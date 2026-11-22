from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from fakenews.predict import load_pipeline, predict_texts

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_pipeline()  # fail fast at startup if the model file is missing
    yield


app = FastAPI(
    title="Fake News Detection API",
    version="0.1.0",
    description=(
        "Classifies a news article as real or fake. Trained on the WELFake dataset "
        "(mostly 2016-2018 US political news), so it recognises patterns of this dataset's "
        "sources, not truth. Do not use it to decide what is factually correct."
    ),
    lifespan=lifespan,
)


class Article(BaseModel):
    text: str = Field(..., min_length=30, max_length=50_000)


class Batch(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=32)


class Prediction(BaseModel):
    label: str
    fake_probability: float
    confidence: float
    tokens_used: int
    warning: str | None = None


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(article: Article):
    return predict_texts([article.text])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: Batch):
    return predict_texts(batch.texts)