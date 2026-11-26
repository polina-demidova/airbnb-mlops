import os
import pickle
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = os.getenv(
    "MODEL_PATH", "/models/model.pkl"
)


class Listing(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accommodates: int = Field(ge=1)
    bathrooms: float = Field(ge=0)
    bedrooms: float = Field(ge=0)
    beds: float = Field(ge=0)
    minimum_nights: int = Field(ge=1)
    neighbourhood_cleansed: str
    property_type: str
    room_type: str


class PredictionResponse(BaseModel):
    predicted_price: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка обученного Pipeline при старте API."""

    with open(MODEL_PATH, "rb") as file:
        app.state.pipeline = pickle.load(file)

    print(f"Model loaded: {MODEL_PATH}")

    yield


app = FastAPI(
    title="Airbnb Price Prediction API",
    description="Prediction of Airbnb listing prices in London",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "service": "Airbnb Price Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "pipeline")
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(listing: Listing):
    """Предсказание стоимости жилья."""

    X = pd.DataFrame([listing.model_dump()])

    prediction = app.state.pipeline.predict(X)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }