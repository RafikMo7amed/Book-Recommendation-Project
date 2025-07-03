from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import json
import logging
from typing import List
import traceback

from . import config
from .summarization_model_handler import SummarizationModelHandler
from . import recommendation_logic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = FastAPI(title="BookWise API v2.0 - Final Version")
state = {}

@app.on_event("startup")
def load_all():
    logger.info("API Server starting up...")
    df = pd.read_json(config.CLASSIFIED_BOOKS_PATH)
    if 'book_id' not in df.columns:
        df['book_id'] = range(len(df))
    df.set_index('book_id', inplace=True)
    state['df_classified'] = df
    
    with open(config.BEST_PARAMS_PATH, 'r') as f:
        state['best_summary_params'] = json.load(f)
        
    state['summarizer'] = SummarizationModelHandler()
    logger.info("Startup complete.")


class UserPreferences(BaseModel):
    goals: List[str] = []; skills: List[str] = []; content_types: List[str] = []; habit_building: bool = False

class SummarizationRequest(BaseModel):
    book_id: int
    reading_time: str = Field(..., pattern=r"^(5 minutes|10 minutes|15\+ minutes)$")

# --- Final API Endpoints ---

@app.get("/recommendations/top-rated", summary="Get top 7 general book recommendations")
def get_top_rated_endpoint():
    df = state.get('df_classified')
    if df is None or df.empty: raise HTTPException(503, "Service not ready.")
    top_rated_df = recommendation_logic.get_top_rated_books(df.copy())
    response = [{"book_id": int(i), "title": r['title'], "cover_url": r.get('cover_url', '')} for i, r in top_rated_df.iterrows()]
    return response

@app.post("/recommendations/for-you", summary="Get 7 personalized recommendations for the user")
def get_for_you_endpoint(preferences: UserPreferences):
    df = state.get('df_classified')
    if df is None or df.empty: raise HTTPException(503, "Service not ready.")
    recommended_df = recommendation_logic.get_for_you_recommendations(df.copy(), preferences.dict())
    response = [{"book_id": int(i), "title": r['title'], "cover_url": r.get('cover_url', '')} for i, r in recommended_df.iterrows()]
    return response

@app.post("/summary", summary="Get an on-demand summary for a single book")
def get_summary_endpoint(request: SummarizationRequest):
    df = state.get('df_classified')
    summarizer = state.get('summarizer')
    best_params = state.get('best_summary_params')

    if df is None or summarizer is None:
        raise HTTPException(503, "Service not ready.")

    try:
        content = df.loc[request.book_id, 'content']
    except KeyError:
        raise HTTPException(404, "Book ID not found.")

    try:
        reading_time_map = {'5 minutes': 0.3, '10 minutes': 0.5, '15\+ minutes': 0.7}
        ratio = reading_time_map.get(request.reading_time)
        
        summary = summarizer.summarize_text(content, params=best_params, ratio=ratio)
        
        if "Error" in summary:
            print(f"--- Summarizer Function Returned an Error: {summary} ---")
            raise HTTPException(500, "Failed to generate summary.")
            
        return {"book_id": request.book_id, "summary": summary}

    except Exception as e:
        
        print("--- UNEXPECTED ERROR TRACEBACK ---")
        print(traceback.format_exc())
        print("----------------------------------")
        raise HTTPException(500, detail="Failed to generate summary due to an internal error.")
