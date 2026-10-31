"""REST API.   uvicorn langscout.api:app --reload      then open http://127.0.0.1:8000/docs"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .detector import LanguageDetector

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["detector"] = LanguageDetector()      # load the model once at start-up
    yield
    state.clear()


app = FastAPI(title="LangScout API", version=__version__,
              description="Language detection for 17 languages plus Hinglish (Hindi in English letters).",
              lifespan=lifespan)


class DetectRequest(BaseModel):
    text: str = Field(..., max_length=20000, examples=["Ceci est une belle journée"])
    top_k: int = Field(3, ge=1, le=10)


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)


class ExplainRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    top_n: int = Field(8, ge=1, le=25)


@app.get("/health")
def health():
    meta = state["detector"].metadata
    return {"status": "ok", "model": meta.get("model"), "trained_on": meta.get("trained_on"), "version": __version__}


@app.get("/languages")
def languages():
    return {"languages": state["detector"].languages}


@app.post("/detect")
def detect(req: DetectRequest):
    return state["detector"].detect(req.text, top_k=req.top_k).as_dict()


@app.post("/detect/batch")
def detect_batch(req: BatchRequest):
    return {"results": [p.as_dict() for p in state["detector"].detect_batch(req.texts, top_k=1)]}


@app.post("/detect/segments")
def detect_segments(req: DetectRequest):
    return {"segments": [p.as_dict() for p in state["detector"].detect_segments(req.text)]}


@app.post("/explain")
def explain(req: ExplainRequest):
    result = state["detector"].explain(req.text, top_n=req.top_n)
    if result is None:
        raise HTTPException(status_code=422, detail="No language could be predicted for this text, so there is nothing to explain.")
    return result
