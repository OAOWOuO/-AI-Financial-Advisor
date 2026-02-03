"""Run history endpoint for storing and retrieving past analysis runs."""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
import os

from app.backend.database import get_db
from app.backend.database.models import Base
from sqlalchemy import Column, Integer, String, Text, DateTime

router = APIRouter(prefix="/run-history")


# Database model for run history
from sqlalchemy import Column, Integer, String, Text, DateTime
from app.backend.database.models import Base

class RunHistory(Base):
    __tablename__ = "run_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    tickers = Column(String(500), nullable=False)  # Comma-separated
    analysts = Column(Text, nullable=True)  # JSON list
    markdown_content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)  # JSON summary data


# Pydantic models
class RunHistoryCreate(BaseModel):
    name: str
    tickers: List[str]
    analysts: Optional[List[str]] = None
    markdown_content: str
    summary: Optional[dict] = None


class RunHistoryResponse(BaseModel):
    id: int
    name: str
    timestamp: str
    tickers: List[str]
    analysts: Optional[List[str]]

    class Config:
        from_attributes = True


class RunHistoryDetail(RunHistoryResponse):
    markdown_content: str
    summary: Optional[dict]


# Create tables
from app.backend.database.connection import engine
Base.metadata.create_all(bind=engine)


@router.get("/", response_model=List[RunHistoryResponse])
async def list_runs(db: Session = Depends(get_db)):
    """List all saved runs (without full content)."""
    runs = db.query(RunHistory).order_by(RunHistory.timestamp.desc()).all()
    return [
        RunHistoryResponse(
            id=run.id,
            name=run.name,
            timestamp=run.timestamp.isoformat(),
            tickers=run.tickers.split(","),
            analysts=json.loads(run.analysts) if run.analysts else None,
        )
        for run in runs
    ]


@router.get("/{run_id}", response_model=RunHistoryDetail)
async def get_run(run_id: int, db: Session = Depends(get_db)):
    """Get a specific run with full content."""
    run = db.query(RunHistory).filter(RunHistory.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return RunHistoryDetail(
        id=run.id,
        name=run.name,
        timestamp=run.timestamp.isoformat(),
        tickers=run.tickers.split(","),
        analysts=json.loads(run.analysts) if run.analysts else None,
        markdown_content=run.markdown_content,
        summary=json.loads(run.summary) if run.summary else None,
    )


@router.post("/", response_model=RunHistoryResponse)
async def create_run(run_data: RunHistoryCreate, db: Session = Depends(get_db)):
    """Save a new run to history."""
    run = RunHistory(
        name=run_data.name,
        tickers=",".join(run_data.tickers),
        analysts=json.dumps(run_data.analysts) if run_data.analysts else None,
        markdown_content=run_data.markdown_content,
        summary=json.dumps(run_data.summary) if run_data.summary else None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    return RunHistoryResponse(
        id=run.id,
        name=run.name,
        timestamp=run.timestamp.isoformat(),
        tickers=run.tickers.split(","),
        analysts=json.loads(run.analysts) if run.analysts else None,
    )


@router.delete("/{run_id}")
async def delete_run(run_id: int, db: Session = Depends(get_db)):
    """Delete a run from history."""
    run = db.query(RunHistory).filter(RunHistory.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    db.delete(run)
    db.commit()
    return {"success": True}


@router.post("/import-file")
async def import_from_file(db: Session = Depends(get_db)):
    """Import the current project1_run.md file as a saved run."""
    # Try to find the markdown file
    possible_paths = [
        "app/frontend/public/project1_run.md",
        "../frontend/public/project1_run.md",
        "public/project1_run.md",
    ]

    content = None
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                content = f.read()
            break

    if not content:
        # Try absolute path
        import pathlib
        base_path = pathlib.Path(__file__).parent.parent.parent.parent
        file_path = base_path / "frontend" / "public" / "project1_run.md"
        if file_path.exists():
            with open(file_path, "r") as f:
                content = f.read()

    if not content:
        raise HTTPException(status_code=404, detail="project1_run.md not found")

    # Parse tickers from content
    import re
    ticker_matches = re.findall(r"Analysis for (\w+)", content)
    tickers = list(set(ticker_matches)) if ticker_matches else ["UNKNOWN"]

    # Create run entry
    run = RunHistory(
        name=f"Project 1 Run - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        tickers=",".join(tickers),
        analysts=None,
        markdown_content=content,
        summary=None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    return RunHistoryResponse(
        id=run.id,
        name=run.name,
        timestamp=run.timestamp.isoformat(),
        tickers=tickers,
        analysts=None,
    )
