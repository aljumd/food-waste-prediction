"""
Food Waste Level Prediction Application & API
Provides real-time machine learning prediction of food waste levels (Low, Medium, High)
based on kitchen operational parameters and portion metrics.
"""

import os
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field, model_validator, field_validator
from sklearn.base import BaseEstimator, TransformerMixin
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# =====================================================================
# 1. DOMAIN FEATURE ENGINEERING TRANSFORMER
# =====================================================================

PORTION_BENCHMARKS = {
    "Breakfast": 0.41,
    "Lunch": 0.58,
    "Dinner": 0.74,
    "Buffet / Event": 0.88,
    "Buffet": 0.88,
    "Event": 0.88
}

class FoodWasteFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature transformer:
    - prepared_per_person: food_prepared_kg / meals_served (kg per diner)
    - prev_waste_ratio: previous_waste_kg / food_prepared_kg (historical waste burden)
    - portion_excess_ratio: (prepared_per_person - baseline) / baseline
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if not isinstance(X_copy, pd.DataFrame):
            raise ValueError("Input to FoodWasteFeatureEngineer must be a pandas DataFrame")

        prepared = X_copy["food_prepared_kg"].values
        served = np.maximum(X_copy["meals_served"].values, 1)
        prev_waste = X_copy["previous_waste_kg"].values

        # 1. Portion per attendee
        prepared_per_person = prepared / served
        X_copy["prepared_per_person"] = np.round(prepared_per_person, 3)

        # 2. Historical waste ratio
        X_copy["prev_waste_ratio"] = np.round(prev_waste / np.maximum(prepared, 0.1), 3)

        # 3. Portion excess relative to typical diner appetite benchmark
        def get_baseline(row):
            mt = row.get("meal_type", "Lunch")
            return PORTION_BENCHMARKS.get(mt, 0.58)

        baselines = np.array([get_baseline(r) for _, r in X_copy.iterrows()])
        X_copy["portion_excess_ratio"] = np.round((prepared_per_person - baselines) / np.maximum(baselines, 0.1), 3)

        return X_copy


# =====================================================================
# 2. PYDANTIC SCHEMAS & INPUT VALIDATION
# =====================================================================

VALID_MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Buffet / Event", "Buffet", "Event"]
VALID_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

class MealPredictionRequest(BaseModel):
    meal_type: str = Field(..., description="Type of meal (Breakfast, Lunch, Dinner, Buffet / Event)", example="Lunch")
    food_prepared_kg: float = Field(..., gt=0, description="Total quantity of food prepared in kg (must be > 0)", example=32.0)
    meals_served: int = Field(..., gt=0, description="Total diners / meals served (must be > 0)", example=48)
    day_of_week: str = Field(..., description="Day of the week (Monday - Sunday)", example="Wednesday")
    previous_waste_kg: Optional[float] = Field(default=None, ge=0, description="Historical waste from previous meal in kg", example=1.5)
    prev_waste_kg: Optional[float] = Field(default=None, ge=0, description="Alias for previous_waste_kg", example=1.5)

    @model_validator(mode="before")
    @classmethod
    def resolve_aliases_and_formats(cls, values: Any) -> Any:
        if isinstance(values, dict):
            # Resolve prev_waste_kg alias
            if "previous_waste_kg" not in values or values["previous_waste_kg"] is None:
                if "prev_waste_kg" in values and values["prev_waste_kg"] is not None:
                    values["previous_waste_kg"] = values["prev_waste_kg"]
                else:
                    values["previous_waste_kg"] = 0.0

            # Canonicalize meal type
            if "meal_type" in values and isinstance(values["meal_type"], str):
                mt = values["meal_type"].strip()
                if mt.lower() in ["buffet", "event", "buffet / event"]:
                    values["meal_type"] = "Buffet / Event"
                elif mt.capitalize() in ["Breakfast", "Lunch", "Dinner"]:
                    values["meal_type"] = mt.capitalize()

            # Canonicalize day of week
            if "day_of_week" in values and isinstance(values["day_of_week"], str):
                values["day_of_week"] = values["day_of_week"].strip().capitalize()

        return values

    @field_validator("meal_type")
    @classmethod
    def validate_meal_type(cls, v: str) -> str:
        if v not in VALID_MEAL_TYPES:
            raise ValueError(f"Invalid meal_type '{v}'. Must be one of {VALID_MEAL_TYPES}")
        return v

    @field_validator("day_of_week")
    @classmethod
    def validate_day_of_week(cls, v: str) -> str:
        if v not in VALID_DAYS:
            raise ValueError(f"Invalid day_of_week '{v}'. Must be one of {VALID_DAYS}")
        return v


class MealPredictionResponse(BaseModel):
    predicted_category: str = Field(..., description="Predicted waste category (Low Waste, Medium Waste, High Waste)")
    confidence: float = Field(..., description="Model probability confidence for winning class (0.0 - 1.0)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities across all 3 waste categories")
    recommendation_message: str = Field(..., description="Actionable kitchen insight")
    efficiency_tag: str = Field(..., description="Status badge tag (High Efficiency, Standard Margin, Surplus Alert)")
    waste_reduction_tips: List[str] = Field(..., description="Contextual waste reduction tips")
    summary: Dict[str, str] = Field(..., description="Echoed inputs for UI display")


# =====================================================================
# 3. MODEL PREDICTOR SINGLETON
# =====================================================================

class WastePredictor:
    def __init__(self, model_file: str = "model.pkl"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, model_file)
        self.pipeline = None
        self.load_model()

    def load_model(self):
        """Loads trained pipeline from model.pkl using joblib or pickle."""
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                print(f"Loaded ML model pipeline from: {self.model_path}")
            except Exception:
                with open(self.model_path, "rb") as f:
                    self.pipeline = pickle.load(f)
                print(f"Loaded ML model pipeline via pickle from: {self.model_path}")
        else:
            print(f"Warning: {self.model_path} not found. Please train the model first.")

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def predict(self, req: MealPredictionRequest) -> MealPredictionResponse:
        if not self.is_loaded:
            self.load_model()
            if not self.is_loaded:
                raise RuntimeError("Model file model.pkl not found or failed to load.")

        input_df = pd.DataFrame([{
            "meal_type": req.meal_type,
            "food_prepared_kg": float(req.food_prepared_kg),
            "meals_served": int(req.meals_served),
            "day_of_week": req.day_of_week,
            "previous_waste_kg": float(req.previous_waste_kg)
        }])

        pred_arr = self.pipeline.predict(input_df)
        predicted_category = str(pred_arr[0])

        probabilities = {}
        confidence = 0.85

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(input_df)[0]
            classes = self.pipeline.classes_
            for cls_name, prob in zip(classes, probs):
                probabilities[str(cls_name)] = round(float(prob), 4)
            confidence = probabilities.get(predicted_category, 0.85)

        portion_per_person = round(req.food_prepared_kg / max(1, req.meals_served), 2)

        if "Low" in predicted_category:
            tag = "High Efficiency 🌱"
            msg = f"The predicted food waste level for this {req.meal_type.lower()} is Low. Food preparation is well-calibrated ({portion_per_person} kg/person)."
            tips = [
                "Maintain current batch cooking portions.",
                "Log unserved leftovers for surplus donation programs.",
                "Keep tracking attendance patterns across weekdays."
            ]
        elif "High" in predicted_category:
            tag = "Surplus Alert ⚠️"
            msg = f"The predicted food waste level for this {req.meal_type.lower()} is High. Preparation quantity ({portion_per_person} kg/person) significantly exceeds estimated consumption."
            tips = [
                f"Reduce batch size by 15-20% for upcoming {req.meal_type.lower()} services.",
                "Prepare food in smaller, staggered batches during peak service windows.",
                "Check shelf-life of prep ingredients and implement FIFO storage protocols.",
                "Review plate waste feedback from yesterday's service."
            ]
        else: # Medium
            tag = "Standard Margin"
            msg = f"The predicted food waste level for this {req.meal_type.lower()} is Medium ({portion_per_person} kg/person)."
            tips = [
                "Monitor real-time customer flow to hold back buffer trays until needed.",
                "Repurpose unserved clean surplus for next-day soups or side preparations.",
                "Calibrate portion spoons to prevent plate over-serving."
            ]

        return MealPredictionResponse(
            predicted_category=predicted_category,
            confidence=round(float(confidence), 3),
            probabilities=probabilities,
            recommendation_message=msg,
            efficiency_tag=tag,
            waste_reduction_tips=tips,
            summary={
                "meal_type": req.meal_type,
                "food_prepared": f"{req.food_prepared_kg} kg",
                "meals_served": str(req.meals_served),
                "day": req.day_of_week,
                "previous_waste": f"{req.previous_waste_kg} kg"
            }
        )


predictor = WastePredictor()


# =====================================================================
# 4. FASTAPI WEB SERVICE & REST ENDPOINTS
# =====================================================================

app = FastAPI(
    title="Food Waste Level Prediction API",
    description="Machine Learning REST service predicting operational food waste categories (Low, Medium, High).",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", summary="Root status check")
def root():
    return {
        "status": "online",
        "service": "Food Waste Level Prediction Engine",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health", summary="Health check")
def health_check():
    return {
        "status": "healthy" if predictor.is_loaded else "degraded",
        "model_loaded": predictor.is_loaded,
        "version": "1.0.0"
    }

@app.get("/model-info", summary="Model metadata")
def model_info():
    return {
        "model_type": "Random Forest Classifier Pipeline",
        "target_classes": ["Low Waste", "Medium Waste", "High Waste"],
        "input_features": [
            "meal_type", "food_prepared_kg", "meals_served", "day_of_week", "previous_waste_kg"
        ],
        "engineered_features": [
            "prepared_per_person", "prev_waste_ratio", "portion_excess_ratio"
        ],
        "thresholds": {
            "low_waste": "<= 12.0% of prepared food",
            "medium_waste": "12.0% - 25.0% of prepared food",
            "high_waste": "> 25.0% of prepared food"
        }
    }

@app.post("/predict", response_model=MealPredictionResponse, summary="Predict waste level")
def predict(request: MealPredictionRequest):
    try:
        return predictor.predict(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


# =====================================================================
# 5. CLI DIRECT EXECUTION
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting Food Waste Prediction server on http://127.0.0.1:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
