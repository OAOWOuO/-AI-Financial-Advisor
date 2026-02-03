"""Alerts endpoint for managing signal alerts."""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json

from app.backend.database import get_db
from app.backend.database.models import Base
from app.backend.database.connection import engine

router = APIRouter(prefix="/alerts")


# Database model
class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    condition_type = Column(String(50), nullable=False)  # 'consensus', 'confidence', 'agent_agrees'
    condition_value = Column(Text, nullable=False)  # JSON with condition details
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_triggered = Column(DateTime, nullable=True)
    trigger_count = Column(Integer, default=0)


# Create table
Base.metadata.create_all(bind=engine)


# Pydantic models
class AlertCondition(BaseModel):
    type: str  # 'consensus', 'confidence', 'agent_agrees'
    # For consensus: {"signal": "BEARISH", "min_count": 5}
    # For confidence: {"min_confidence": 80, "signal": "BULLISH"}
    # For agent_agrees: {"agents": ["warren_buffett", "charlie_munger"], "signal": "BEARISH"}
    params: dict


class AlertCreate(BaseModel):
    name: str
    condition: AlertCondition


class AlertResponse(BaseModel):
    id: int
    name: str
    condition_type: str
    condition_value: dict
    is_active: bool
    created_at: str
    last_triggered: Optional[str]
    trigger_count: int


class AlertCheckResult(BaseModel):
    alert_id: int
    alert_name: str
    triggered: bool
    message: str
    details: Optional[dict]


class TriggeredAlert(BaseModel):
    id: int
    name: str
    message: str
    ticker: Optional[str]
    details: dict


@router.get("/", response_model=List[AlertResponse])
async def list_alerts(db: Session = Depends(get_db)):
    """List all alerts."""
    alerts = db.query(Alert).order_by(Alert.created_at.desc()).all()
    return [
        AlertResponse(
            id=a.id,
            name=a.name,
            condition_type=a.condition_type,
            condition_value=json.loads(a.condition_value),
            is_active=a.is_active,
            created_at=a.created_at.isoformat(),
            last_triggered=a.last_triggered.isoformat() if a.last_triggered else None,
            trigger_count=a.trigger_count,
        )
        for a in alerts
    ]


@router.post("/", response_model=AlertResponse)
async def create_alert(alert_data: AlertCreate, db: Session = Depends(get_db)):
    """Create a new alert."""
    alert = Alert(
        name=alert_data.name,
        condition_type=alert_data.condition.type,
        condition_value=json.dumps(alert_data.condition.params),
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)

    return AlertResponse(
        id=alert.id,
        name=alert.name,
        condition_type=alert.condition_type,
        condition_value=json.loads(alert.condition_value),
        is_active=alert.is_active,
        created_at=alert.created_at.isoformat(),
        last_triggered=None,
        trigger_count=0,
    )


@router.delete("/{alert_id}")
async def delete_alert(alert_id: int, db: Session = Depends(get_db)):
    """Delete an alert."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    db.delete(alert)
    db.commit()
    return {"success": True}


@router.put("/{alert_id}/toggle")
async def toggle_alert(alert_id: int, db: Session = Depends(get_db)):
    """Toggle alert active status."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.is_active = not alert.is_active
    db.commit()
    return {"success": True, "is_active": alert.is_active}


@router.post("/check", response_model=List[TriggeredAlert])
async def check_alerts(analysis_data: dict, db: Session = Depends(get_db)):
    """Check if any alerts are triggered by the given analysis data."""
    triggered = []
    alerts = db.query(Alert).filter(Alert.is_active == True).all()

    # Extract signals from analysis data
    # Expected format: {"stocks": [{"ticker": "AAPL", "agents": [{"agent": "...", "signal": "...", "confidence": ...}]}]}
    stocks = analysis_data.get("stocks", [])
    portfolio = analysis_data.get("portfolio", [])

    for alert in alerts:
        condition = json.loads(alert.condition_value)

        if alert.condition_type == "consensus":
            # Check if X or more agents agree on a signal
            required_signal = condition.get("signal", "BEARISH").upper()
            min_count = condition.get("min_count", 5)

            for stock in stocks:
                signal_count = sum(
                    1 for a in stock.get("agents", [])
                    if a.get("signal", "").upper() == required_signal
                )
                if signal_count >= min_count:
                    triggered.append(TriggeredAlert(
                        id=alert.id,
                        name=alert.name,
                        message=f"{signal_count} agents are {required_signal} on {stock['ticker']}",
                        ticker=stock["ticker"],
                        details={"signal_count": signal_count, "required": min_count},
                    ))
                    # Update alert trigger info
                    alert.last_triggered = datetime.utcnow()
                    alert.trigger_count += 1

        elif alert.condition_type == "confidence":
            # Check if average confidence exceeds threshold for a signal
            min_confidence = condition.get("min_confidence", 80)
            target_signal = condition.get("signal", "").upper()

            for stock in stocks:
                matching_agents = [
                    a for a in stock.get("agents", [])
                    if not target_signal or a.get("signal", "").upper() == target_signal
                ]
                if matching_agents:
                    avg_confidence = sum(a.get("confidence", 0) for a in matching_agents) / len(matching_agents)
                    if avg_confidence >= min_confidence:
                        triggered.append(TriggeredAlert(
                            id=alert.id,
                            name=alert.name,
                            message=f"High confidence ({avg_confidence:.0f}%) {target_signal or 'signal'} on {stock['ticker']}",
                            ticker=stock["ticker"],
                            details={"avg_confidence": avg_confidence, "threshold": min_confidence},
                        ))
                        alert.last_triggered = datetime.utcnow()
                        alert.trigger_count += 1

        elif alert.condition_type == "agent_agrees":
            # Check if specific agents all agree on a signal
            required_agents = set(a.lower() for a in condition.get("agents", []))
            target_signal = condition.get("signal", "").upper()

            for stock in stocks:
                agent_signals = {
                    a.get("agent", "").lower().replace(" ", "_"): a.get("signal", "").upper()
                    for a in stock.get("agents", [])
                }
                matching = [
                    agent for agent in required_agents
                    if any(agent in k for k in agent_signals.keys())
                    and agent_signals.get(next((k for k in agent_signals if agent in k), ""), "") == target_signal
                ]
                if len(matching) == len(required_agents) and required_agents:
                    triggered.append(TriggeredAlert(
                        id=alert.id,
                        name=alert.name,
                        message=f"Selected agents agree: {target_signal} on {stock['ticker']}",
                        ticker=stock["ticker"],
                        details={"agents": list(matching), "signal": target_signal},
                    ))
                    alert.last_triggered = datetime.utcnow()
                    alert.trigger_count += 1

    db.commit()
    return triggered


# Preset alert templates
@router.get("/templates")
async def get_alert_templates():
    """Get preset alert templates."""
    return {
        "templates": [
            {
                "name": "Strong Bearish Consensus",
                "description": "Alert when 5+ agents are bearish on a stock",
                "condition": {"type": "consensus", "params": {"signal": "BEARISH", "min_count": 5}},
            },
            {
                "name": "Strong Bullish Consensus",
                "description": "Alert when 5+ agents are bullish on a stock",
                "condition": {"type": "consensus", "params": {"signal": "BULLISH", "min_count": 5}},
            },
            {
                "name": "High Confidence Signal",
                "description": "Alert when average confidence exceeds 80%",
                "condition": {"type": "confidence", "params": {"min_confidence": 80}},
            },
            {
                "name": "Value Investors Agree (Bearish)",
                "description": "Alert when Buffett, Munger, and Graham all bearish",
                "condition": {"type": "agent_agrees", "params": {"agents": ["warren_buffett", "charlie_munger", "ben_graham"], "signal": "BEARISH"}},
            },
            {
                "name": "Value Investors Agree (Bullish)",
                "description": "Alert when Buffett, Munger, and Graham all bullish",
                "condition": {"type": "agent_agrees", "params": {"agents": ["warren_buffett", "charlie_munger", "ben_graham"], "signal": "BULLISH"}},
            },
        ]
    }
