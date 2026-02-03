"""Simple analysis endpoint that actually runs the hedge fund analysis."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import asyncio
import json
import os
import sys

router = APIRouter(prefix="/simple-analysis")

# Store for current run status
current_run = {
    "status": "idle",  # idle, running, complete, error
    "progress": [],
    "result": None,
    "error": None,
}


class AnalysisRequest(BaseModel):
    tickers: List[str]
    analysts: List[str]
    model_name: str = "gpt-4o-mini"


@router.get("/status")
async def get_status():
    """Get current run status."""
    return current_run


@router.post("/clear")
async def clear_status():
    """Clear current run status."""
    global current_run
    current_run = {"status": "idle", "progress": [], "result": None, "error": None}
    return {"success": True}


@router.post("/run")
async def run_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start a new analysis run."""
    global current_run

    if current_run["status"] == "running":
        raise HTTPException(status_code=400, detail="Analysis already running")

    # Reset status
    current_run = {
        "status": "running",
        "progress": [f"Starting analysis for {', '.join(request.tickers)}..."],
        "result": None,
        "error": None,
    }

    # Run in background
    background_tasks.add_task(
        run_analysis_task,
        request.tickers,
        request.analysts,
        request.model_name,
    )

    return {"status": "started"}


async def run_analysis_task(tickers: List[str], analysts: List[str], model_name: str):
    """Background task to run the analysis."""
    global current_run

    try:
        current_run["progress"].append("Loading environment...")

        # Set up environment
        from dotenv import load_dotenv
        load_dotenv()

        # Check for API keys
        if not os.environ.get("OPENAI_API_KEY"):
            # Try to load from database
            try:
                from app.backend.database.connection import SessionLocal
                from app.backend.services.api_key_service import ApiKeyService
                db = SessionLocal()
                api_service = ApiKeyService(db)
                keys = api_service.get_api_keys_dict()
                if keys.get("openai"):
                    os.environ["OPENAI_API_KEY"] = keys["openai"]
                if keys.get("financial_datasets"):
                    os.environ["FINANCIAL_DATASETS_API_KEY"] = keys["financial_datasets"]
                db.close()
            except Exception as e:
                current_run["progress"].append(f"Warning: Could not load API keys: {e}")

        current_run["progress"].append("Importing hedge fund modules...")

        # Import the main function
        from src.main import run_hedge_fund, create_workflow
        from src.utils.progress import progress

        # Set up progress callback
        def progress_callback(agent_name, ticker, status, analysis, timestamp):
            msg = f"{agent_name}"
            if ticker:
                msg += f" [{ticker}]"
            if status:
                msg += f": {status}"
            current_run["progress"].append(msg)

        progress.register_handler(progress_callback)

        current_run["progress"].append("Building portfolio...")

        # Calculate dates
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        # Create portfolio
        portfolio = {
            "cash": 100000,
            "margin_requirement": 0.0,
            "margin_used": 0.0,
            "positions": {
                ticker: {
                    "long": 0,
                    "short": 0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                    "short_margin_used": 0.0,
                }
                for ticker in tickers
            },
            "realized_gains": {
                ticker: {"long": 0.0, "short": 0.0}
                for ticker in tickers
            },
        }

        current_run["progress"].append(f"Running analysis with {len(analysts)} analysts...")

        # Run the hedge fund
        result = run_hedge_fund(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=True,
            selected_analysts=analysts,
            model_name=model_name,
            model_provider="OpenAI",
        )

        progress.unregister_handler(progress_callback)

        current_run["progress"].append("Analysis complete!")
        current_run["result"] = {
            "decisions": result.get("decisions"),
            "analyst_signals": result.get("analyst_signals"),
            "tickers": tickers,
            "timestamp": datetime.now().isoformat(),
        }
        current_run["status"] = "complete"

    except Exception as e:
        import traceback
        current_run["error"] = str(e)
        current_run["progress"].append(f"Error: {str(e)}")
        current_run["status"] = "error"
        traceback.print_exc()


@router.get("/result")
async def get_result():
    """Get the result of the last run."""
    if current_run["status"] != "complete":
        raise HTTPException(status_code=400, detail=f"No complete result. Status: {current_run['status']}")

    return current_run["result"]
