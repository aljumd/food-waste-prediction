"""
Food Waste Level Prediction Application & API
Provides real-time machine learning prediction of food waste levels (Low, Medium, High)
based on kitchen operational parameters and portion metrics.
Serves the interactive mobile web UI directly at http://127.0.0.1:8000.
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
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

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

def get_ui_html() -> str:
    """Returns the complete frontend web UI."""
    local_path = os.path.join(os.path.dirname(__file__), "..", "food-waste-ui", "index.html")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    return EMBEDDED_UI_HTML

@app.get("/", summary="Interactive Web UI")
def root(request: Request):
    """
    Serves the complete interactive Food Waste Prediction Mobile Web UI.
    If requested with 'Accept: application/json', returns the API status catalog.
    """
    accept_header = request.headers.get("accept", "")
    if "application/json" in accept_header and "text/html" not in accept_header:
        return JSONResponse(content={
            "status": "online",
            "service": "Food Waste Level Prediction Engine",
            "endpoints": {
                "ui": "/",
                "health": "/health",
                "model_info": "/model-info",
                "predict": "/predict",
                "docs": "/docs"
            }
        })
    return HTMLResponse(content=get_ui_html())

@app.get("/api", summary="Root API status catalog")
def api_catalog():
    return {
        "status": "online",
        "service": "Food Waste Level Prediction Engine",
        "endpoints": {
            "ui": "/",
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
# 5. EMBEDDED WEB UI HTML
# =====================================================================

EMBEDDED_UI_HTML = '<!DOCTYPE html>\n<html lang="en">\n<head>\n  <meta charset="UTF-8" />\n  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />\n  <title>Food Waste Level Prediction</title>\n  \n  <!-- Premium Typography: Plus Jakarta Sans -->\n  <link rel="preconnect" href="https://fonts.googleapis.com" />\n  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />\n  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400;1,600&display=swap" rel="stylesheet" />\n  \n  <!-- Tailwind CSS CDN -->\n  <script src="https://cdn.tailwindcss.com"></script>\n  \n  <!-- Lucide Icons -->\n  <script src="https://unpkg.com/lucide@latest"></script>\n  \n  <!-- Canvas Confetti -->\n  <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>\n\n  <script>\n    tailwind.config = {\n      theme: {\n        extend: {\n          fontFamily: {\n            sans: [\'"Plus Jakarta Sans"\', \'-apple-system\', \'BlinkMacSystemFont\', \'sans-serif\'],\n          },\n          colors: {\n            brand: {\n              dark: \'#1c4a35\',       // Primary dark green for buttons & titles\n              darker: \'#133526\',     // Hover/active state\n              emerald: \'#296b4f\',    // Subheading green\n              light: \'#f2f8f4\',      // Soft background tint\n              mint: \'#e0efe6\',       // Icon circle background\n              border: \'#e4ece6\',     // Card border\n              muted: \'#628072\',      // Secondary text\n            },\n            waste: {\n              cream: \'#fdf7e3\',      // Medium waste card background\n              creamBorder: \'#faecc5\',// Medium waste card border\n              amber: \'#b76918\',      // Medium waste title\n              amberIcon: \'#d97706\',  // Medium waste badge icon\n              \n              greenBg: \'#eef8f2\',    // Low waste card bg\n              greenBorder: \'#d0ebd8\',// Low waste border\n              greenText: \'#236b47\',  // Low waste title\n              greenIcon: \'#2e8b57\',  // Low waste icon\n              \n              redBg: \'#fef1f0\',      // High waste card bg\n              redBorder: \'#fcd3cf\',  // High waste border\n              redText: \'#c23b2b\',    // High waste title\n              redIcon: \'#e04f3d\',    // High waste icon\n            }\n          },\n          boxShadow: {\n            \'phone\': \'0 30px 70px -15px rgba(10, 30, 20, 0.4), 0 0 0 1px rgba(255,255,255,0.1), 0 0 0 8px #142018\',\n            \'subtle\': \'0 2px 8px -2px rgba(20, 50, 35, 0.05)\',\n            \'pill\': \'0 6px 18px -4px rgba(28, 74, 53, 0.3)\',\n          }\n        }\n      }\n    }\n  </script>\n\n  <style>\n    * {\n      box-sizing: border-box;\n      -webkit-font-smoothing: antialiased;\n      -moz-osx-font-smoothing: grayscale;\n    }\n\n    body {\n      font-family: \'Plus Jakarta Sans\', sans-serif;\n      -webkit-tap-highlight-color: transparent;\n      user-select: none;\n      background: radial-gradient(circle at 50% 20%, #172a22 0%, #0d1712 100%);\n      color: #1c4a35;\n    }\n\n    /* Outer Mockup Framing */\n    .iphone-frame {\n      width: 382px;\n      height: 792px;\n      background: #111a14;\n      border-radius: 54px;\n      padding: 10px;\n      box-shadow: 0 35px 80px -20px rgba(0, 0, 0, 0.85), \n                  0 0 0 2px #26382e, \n                  0 0 0 8px #0d1610;\n      position: relative;\n    }\n\n    .iphone-screen {\n      width: 100%;\n      height: 100%;\n      background: #f4f8f5;\n      border-radius: 44px;\n      overflow: hidden;\n      position: relative;\n      display: flex;\n      flex-direction: column;\n    }\n\n    /* Screen Slide Animations */\n    .app-screen {\n      position: absolute;\n      inset: 0;\n      width: 100%;\n      height: 100%;\n      transition: transform 0.38s cubic-bezier(0.32, 0.72, 0, 1), opacity 0.35s ease;\n      display: flex;\n      flex-direction: column;\n      background: #f4f8f5;\n      overflow-y: auto;\n      overflow-x: hidden;\n      scrollbar-width: none;\n    }\n    .app-screen::-webkit-scrollbar {\n      display: none;\n    }\n\n    .screen-active {\n      transform: translateX(0%);\n      opacity: 1;\n      pointer-events: auto;\n      z-index: 10;\n    }\n    .screen-prev {\n      transform: translateX(-100%);\n      opacity: 0;\n      pointer-events: none;\n      z-index: 5;\n    }\n    .screen-next {\n      transform: translateX(100%);\n      opacity: 0;\n      pointer-events: none;\n      z-index: 5;\n    }\n\n    /* Dynamic Island */\n    .dynamic-island {\n      width: 100px;\n      height: 24px;\n      background: #000;\n      border-radius: 18px;\n      position: absolute;\n      top: 10px;\n      left: 50%;\n      transform: translateX(-50%);\n      z-index: 50;\n      display: flex;\n      align-items: center;\n      justify-content: space-between;\n      padding: 0 10px;\n    }\n\n    /* Form Input Cards */\n    .app-input-row {\n      background-color: #ffffff;\n      border: 1.2px solid #e3ede6;\n      border-radius: 18px;\n      padding: 9px 14px;\n      display: flex;\n      align-items: center;\n      gap: 13px;\n      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);\n      box-shadow: 0 1px 3px rgba(24, 60, 42, 0.03);\n    }\n    .app-input-row:focus-within {\n      border-color: #1c4a35;\n      box-shadow: 0 0 0 3px rgba(28, 74, 53, 0.12);\n      transform: translateY(-1px);\n    }\n\n    .icon-bubble {\n      width: 38px;\n      height: 38px;\n      border-radius: 50%;\n      background-color: #e2efe7;\n      color: #1c4a35;\n      display: flex;\n      align-items: center;\n      justify-content: center;\n      flex-shrink: 0;\n    }\n\n    /* Organic Floating Foliage Animations */\n    @keyframes gentleFloat1 {\n      0%, 100% { transform: translateY(0px) rotate(0deg); }\n      50% { transform: translateY(-7px) rotate(3deg); }\n    }\n    @keyframes gentleFloat2 {\n      0%, 100% { transform: translateY(0px) rotate(0deg); }\n      50% { transform: translateY(6px) rotate(-3deg); }\n    }\n    .float-leaf-1 { animation: gentleFloat1 7s ease-in-out infinite; }\n    .float-leaf-2 { animation: gentleFloat2 8s ease-in-out infinite; }\n\n    /* Button Pulse on click */\n    .btn-primary {\n      background: #1c4a35;\n      color: #ffffff;\n      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);\n    }\n    .btn-primary:hover {\n      background: #143727;\n      box-shadow: 0 8px 22px -4px rgba(20, 55, 39, 0.4);\n    }\n    .btn-primary:active {\n      transform: scale(0.98);\n    }\n\n    /* Ambient Soft Glow Behind Bowl */\n    .glow-backdrop {\n      background: radial-gradient(circle, rgba(162, 219, 185, 0.45) 0%, rgba(244, 248, 245, 0) 70%);\n    }\n  </style>\n</head>\n<body class="min-h-screen flex flex-col items-center justify-between p-3 sm:p-5 text-brand-dark">\n\n  <!-- ========================================================================= -->\n  <!-- TOP TOOLBAR: Mode Switcher & Live Controls -->\n  <!-- ========================================================================= -->\n  <header class="w-full max-w-5xl mb-4 flex flex-wrap items-center justify-between gap-3 px-5 py-3 bg-slate-800/80 backdrop-blur-xl rounded-2xl border border-slate-700/70 text-white shadow-2xl">\n    <div class="flex items-center gap-3">\n      <div class="w-9 h-9 rounded-xl bg-emerald-700/90 flex items-center justify-center text-emerald-100 shadow-md ring-1 ring-emerald-500/30">\n        <i data-lucide="leaf" class="w-5 h-5"></i>\n      </div>\n      <div>\n        <h1 class="text-sm font-bold text-slate-100 tracking-tight">Food Waste Level Prediction</h1>\n        <p class="text-[11px] text-slate-400 font-medium">Polished Mobile Application & Prototype</p>\n      </div>\n    </div>\n\n    <!-- Mode Switcher -->\n    <div class="flex items-center bg-slate-900/90 p-1 rounded-xl border border-slate-700/80 text-xs">\n      <button id="tabInteractive" onclick="switchMainView(\'interactive\')" class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition-all bg-emerald-700 text-white shadow-sm">\n        <i data-lucide="smartphone" class="w-3.5 h-3.5"></i>\n        <span>Interactive Mobile</span>\n      </button>\n      <button id="tabShowcase" onclick="switchMainView(\'showcase\')" class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition-all text-slate-300 hover:text-white">\n        <i data-lucide="layout-grid" class="w-3.5 h-3.5"></i>\n        <span>3-Screen Showcase</span>\n      </button>\n    </div>\n\n    <!-- Quick Scenario Presets -->\n    <div class="flex items-center gap-1.5 text-xs">\n      <span class="text-slate-400 text-[11px] font-medium hidden sm:inline">Scenario:</span>\n      <button onclick="setScenario(\'medium\')" class="px-2.5 py-1 bg-amber-950/70 text-amber-200 border border-amber-600/40 rounded-lg hover:bg-amber-900 transition font-semibold">Medium (Mockup)</button>\n      <button onclick="setScenario(\'low\')" class="px-2.5 py-1 bg-emerald-950/70 text-emerald-200 border border-emerald-600/40 rounded-lg hover:bg-emerald-900 transition font-semibold">Low</button>\n      <button onclick="setScenario(\'high\')" class="px-2.5 py-1 bg-rose-950/70 text-rose-200 border border-rose-600/40 rounded-lg hover:bg-rose-900 transition font-semibold">High</button>\n    </div>\n  </header>\n\n\n  <!-- ========================================================================= -->\n  <!-- 1. INTERACTIVE PHONE SIMULATOR VIEW (Default) -->\n  <!-- ========================================================================= -->\n  <main id="viewInteractiveContainer" class="flex items-center justify-center my-auto transition-all">\n    <div class="iphone-frame">\n      <div class="iphone-screen shadow-inner select-none">\n        \n        <!-- STATUS BAR (iOS Style 44px) -->\n        <div class="h-11 w-full shrink-0 flex items-center justify-between px-7 pt-1.5 text-brand-dark text-[11.5px] font-bold z-40 relative select-none">\n          <span id="liveClock" class="tracking-tight font-extrabold text-[12px]">9:41</span>\n          \n          <div class="dynamic-island">\n            <div class="w-2.5 h-2.5 rounded-full bg-[#181818]"></div>\n            <div class="w-2.5 h-2.5 rounded-full bg-[#0d2a1c] ring-1 ring-emerald-500/60"></div>\n          </div>\n\n          <div class="flex items-center gap-1.5 text-brand-dark opacity-90">\n            <!-- Cellular Signal -->\n            <svg class="w-3.5 h-3 fill-current" viewBox="0 0 24 24"><path d="M2 17h3v4H2v-4zm5-4h3v8H7v-8zm5-4h3v12h-3V9zm5-4h3v16h-3V5z"/></svg>\n            <!-- Wifi -->\n            <svg class="w-3.5 h-3 fill-current" viewBox="0 0 24 24"><path d="M12 4C7.31 4 3.07 5.9 0 8.98L12 21 24 8.98C20.93 5.9 16.69 4 12 4zm0 3.5c3.7 0 7.07 1.48 9.55 3.9L12 19.42 2.45 11.4C4.93 8.98 8.3 7.5 12 7.5z"/></svg>\n            <!-- Battery Icon -->\n            <div class="w-4.5 h-2.5 border-[1.2px] border-current rounded-[3px] p-[1.5px] flex items-center">\n              <div class="h-full bg-current w-3/4 rounded-[1px]"></div>\n            </div>\n          </div>\n        </div>\n\n        <!-- APP CONTAINER -->\n        <div class="flex-1 relative overflow-hidden">\n\n          <!-- ============================================================ -->\n          <!-- SCREEN 1: ONBOARDING / WELCOME -->\n          <!-- ============================================================ -->\n          <section id="scr1" class="app-screen screen-active p-5 pt-0 pb-4 justify-between">\n            \n            <!-- Soft Organic Decorative Floating Leaves -->\n            <div class="absolute -top-3 -left-3 w-28 h-28 pointer-events-none float-leaf-1 opacity-70">\n              <svg viewBox="0 0 100 100" class="w-full h-full fill-emerald-600/15">\n                <path d="M20,60 C10,40 30,10 60,10 C80,30 90,60 70,80 C50,90 30,80 20,60 Z"/>\n              </svg>\n            </div>\n            <div class="absolute top-16 -right-5 w-28 h-28 pointer-events-none float-leaf-2 opacity-60">\n              <svg viewBox="0 0 100 100" class="w-full h-full fill-emerald-600/15">\n                <path d="M30,10 C60,20 80,40 70,70 C50,90 20,80 10,50 C10,30 20,20 30,10 Z"/>\n              </svg>\n            </div>\n            <div class="absolute bottom-16 -left-6 w-32 h-32 pointer-events-none float-leaf-1 opacity-60">\n              <svg viewBox="0 0 100 100" class="w-full h-full fill-emerald-700/15">\n                <path d="M10,80 C10,50 30,20 60,20 C80,40 80,70 60,90 C40,100 20,90 10,80 Z"/>\n              </svg>\n            </div>\n            <div class="absolute bottom-3 -right-4 w-28 h-28 pointer-events-none float-leaf-2 opacity-70">\n              <svg viewBox="0 0 100 100" class="w-full h-full fill-emerald-600/15">\n                <path d="M20,20 C50,10 80,30 80,60 C70,90 40,90 20,70 C10,50 10,30 20,20 Z"/>\n              </svg>\n            </div>\n\n            <!-- Top Header & Logo -->\n            <div class="flex flex-col items-center text-center mt-1 z-10">\n              <!-- Leaf App Icon -->\n              <div class="w-13 h-13 p-3 bg-brand-dark rounded-2xl flex items-center justify-center text-white mb-2 shadow-lg shadow-brand-dark/20 transform -rotate-6 transition-transform hover:rotate-0">\n                <i data-lucide="leaf" class="w-7 h-7 text-emerald-300"></i>\n              </div>\n              \n              <h1 class="text-[25px] font-black text-brand-dark tracking-tight leading-[1.15]">\n                Food Waste<br />\n                <span class="text-brand-emerald text-[22px] font-extrabold">Level Prediction</span>\n              </h1>\n              \n              <p class="text-brand-muted text-xs font-semibold mt-1 max-w-[210px] leading-relaxed">\n                Smarter decisions for<br />less food waste.\n              </p>\n            </div>\n\n            <!-- Salad Bowl Illustration (Polished & Rich SVG) -->\n            <div class="relative my-auto flex items-center justify-center z-10 py-1">\n              <!-- Soft Aura Glow -->\n              <div class="absolute w-60 h-60 glow-backdrop rounded-full pointer-events-none"></div>\n\n              <div class="w-64 h-52 relative flex items-center justify-center">\n                <svg viewBox="0 0 340 280" class="w-full h-full drop-shadow-xl">\n                  <defs>\n                    <!-- Bowl Gradient -->\n                    <linearGradient id="bowlGrad" x1="0%" y1="0%" x2="0%" y2="100%">\n                      <stop offset="0%" stop-color="#245940" />\n                      <stop offset="100%" stop-color="#163e2c" />\n                    </linearGradient>\n                    <!-- Greens Gradient -->\n                    <linearGradient id="leafGrad1" x1="0%" y1="0%" x2="100%" y2="100%">\n                      <stop offset="0%" stop-color="#5ec28b" />\n                      <stop offset="100%" stop-color="#3b8c5f" />\n                    </linearGradient>\n                    <linearGradient id="leafGrad2" x1="0%" y1="0%" x2="100%" y2="100%">\n                      <stop offset="0%" stop-color="#80d5a3" />\n                      <stop offset="100%" stop-color="#49a171" />\n                    </linearGradient>\n                  </defs>\n\n                  <!-- Ambient Floating Foliage Leaves -->\n                  <g opacity="0.95">\n                    <path d="M55 110 C 25 75, 65 30, 115 40 C 100 80, 90 105, 55 110 Z" fill="url(#leafGrad1)" />\n                    <path d="M285 95 C 320 60, 295 20, 245 30 C 255 70, 270 90, 285 95 Z" fill="url(#leafGrad2)" />\n                    <path d="M40 160 C 10 145, 15 110, 55 115 C 50 140, 45 155, 40 160 Z" fill="#8edcb0" />\n                    <path d="M300 160 C 330 145, 325 110, 285 115 C 290 140, 295 155, 300 160 Z" fill="url(#leafGrad1)" />\n                  </g>\n\n                  <!-- Soft Cast Shadow Under Bowl -->\n                  <ellipse cx="170" cy="245" rx="110" ry="14" fill="#143727" opacity="0.18" />\n\n                  <!-- Fresh Greens Base Mound -->\n                  <path d="M65 155 C 55 110, 110 65, 170 70 C 230 65, 285 110, 275 155 Z" fill="#36835a" />\n                  <path d="M85 138 C 105 90, 150 78, 185 82 C 230 78, 260 110, 255 145 Z" fill="url(#leafGrad1)" />\n\n                  <!-- Ingredients -->\n                  <!-- Whole Top Cherry Tomato with stem -->\n                  <circle cx="230" cy="102" r="18" fill="#e74c3c" />\n                  <circle cx="230" cy="102" r="14" fill="#c0392b" />\n                  <circle cx="234" cy="98" r="3.5" fill="#f39c12" />\n                  <path d="M230 84 C 228 80, 233 80, 230 84 Z" stroke="#163e2c" stroke-width="3" stroke-linecap="round" />\n\n                  <!-- Sliced Left Tomato with detail rings -->\n                  <circle cx="115" cy="148" r="22" fill="#e74c3c" />\n                  <circle cx="115" cy="148" r="17" fill="#c0392b" />\n                  <circle cx="115" cy="148" r="10" fill="#e67e22" stroke="#e74c3c" stroke-width="2.5" stroke-dasharray="3,3" />\n\n                  <!-- Hard Boiled Egg with golden yolk -->\n                  <ellipse cx="185" cy="116" rx="20" ry="15" fill="#ffffff" />\n                  <circle cx="185" cy="116" r="9" fill="#f39c12" />\n\n                  <!-- Roasted Grains, Quinoa & Falafel bites -->\n                  <circle cx="145" cy="122" r="16" fill="#d4a373" />\n                  <circle cx="148" cy="124" r="13" fill="#bc6c25" />\n                  <circle cx="170" cy="142" r="15" fill="#e9edc9" />\n                  <circle cx="198" cy="148" r="16" fill="#faedcd" />\n                  <circle cx="245" cy="132" r="15" fill="#dda15e" />\n\n                  <!-- Crisp Cucumber / Avocado slices -->\n                  <circle cx="90" cy="135" r="14" fill="#95d5b2" stroke="#2d6a4f" stroke-width="3" />\n                  <circle cx="90" cy="135" r="7" fill="#d8f3dc" />\n\n                  <!-- Salad Bowl Ceramic Body (Forest Green) -->\n                  <path d="M45 145 C 50 220, 95 245, 170 245 C 245 245, 290 220, 295 145 C 295 136, 45 136, 45 145 Z" fill="url(#bowlGrad)" />\n                  <!-- Bowl Top Rim Highlight -->\n                  <ellipse cx="170" cy="143" rx="125" ry="11" fill="#2d6a4f" />\n                  <!-- Bowl Inner Shadow Rim -->\n                  <ellipse cx="170" cy="143" rx="122" ry="9" fill="#1e4c37" />\n                  <!-- Bowl Bottom Curve Line Accent -->\n                  <path d="M75 185 C 105 225, 235 225, 265 185" stroke="#3fa272" stroke-width="3.5" stroke-linecap="round" fill="none" opacity="0.65" />\n                </svg>\n              </div>\n            </div>\n\n            <!-- 3 Pillars Feature Badges -->\n            <div class="grid grid-cols-3 gap-2 px-0.5 mb-3 z-10">\n              <!-- Feature 1 -->\n              <div class="flex flex-col items-center text-center p-2.5 rounded-2xl bg-white/90 backdrop-blur-sm border border-brand-border shadow-subtle transition-transform hover:-translate-y-0.5">\n                <div class="w-8 h-8 rounded-full bg-brand-mint flex items-center justify-center text-brand-dark mb-1 shadow-xs">\n                  <i data-lucide="leaf" class="w-4 h-4"></i>\n                </div>\n                <span class="text-[9.5px] font-bold text-brand-dark leading-tight">Reduce<br />Food Waste</span>\n              </div>\n\n              <!-- Feature 2 -->\n              <div class="flex flex-col items-center text-center p-2.5 rounded-2xl bg-white/90 backdrop-blur-sm border border-brand-border shadow-subtle transition-transform hover:-translate-y-0.5">\n                <div class="w-8 h-8 rounded-full bg-brand-mint flex items-center justify-center text-brand-dark mb-1 shadow-xs">\n                  <i data-lucide="globe" class="w-4 h-4"></i>\n                </div>\n                <span class="text-[9.5px] font-bold text-brand-dark leading-tight">Save<br />Resources</span>\n              </div>\n\n              <!-- Feature 3 -->\n              <div class="flex flex-col items-center text-center p-2.5 rounded-2xl bg-white/90 backdrop-blur-sm border border-brand-border shadow-subtle transition-transform hover:-translate-y-0.5">\n                <div class="w-8 h-8 rounded-full bg-brand-mint flex items-center justify-center text-brand-dark mb-1 shadow-xs">\n                  <i data-lucide="sprout" class="w-4 h-4"></i>\n                </div>\n                <span class="text-[9.5px] font-bold text-brand-dark leading-tight">Build a<br />Greener Future</span>\n              </div>\n            </div>\n\n            <!-- Bottom CTA Button -->\n            <div class="z-10">\n              <button onclick="navTo(2)" class="w-full btn-primary py-3.5 px-6 rounded-full font-bold text-sm flex items-center justify-center gap-2 shadow-pill">\n                <span>Predict Waste</span>\n                <i data-lucide="arrow-right" class="w-4 h-4"></i>\n              </button>\n              <div class="w-28 h-1 bg-slate-300 rounded-full mx-auto mt-3 opacity-60"></div>\n            </div>\n          </section>\n\n\n          <!-- ============================================================ -->\n          <!-- SCREEN 2: ENTER MEAL DETAILS (INPUT FORM) -->\n          <!-- ============================================================ -->\n          <section id="scr2" class="app-screen screen-next p-5 pt-0 pb-4 justify-between">\n            <div class="absolute bottom-2 -left-4 w-28 h-28 pointer-events-none opacity-20">\n              <svg viewBox="0 0 100 100" class="fill-emerald-800"><path d="M10,80 C10,50 30,20 60,20 C80,40 80,70 60,90 C40,100 20,90 10,80 Z"/></svg>\n            </div>\n\n            <div>\n              <!-- Header Bar -->\n              <div class="flex items-center justify-between mb-2">\n                <button onclick="navTo(1)" class="w-8 h-8 rounded-full bg-white border border-brand-border text-brand-dark flex items-center justify-center hover:bg-brand-mint transition active:scale-95 shadow-xs">\n                  <i data-lucide="arrow-left" class="w-4 h-4"></i>\n                </button>\n                <h2 class="text-sm font-bold text-brand-dark tracking-tight">Enter Meal Details</h2>\n                <div class="w-8"></div>\n              </div>\n\n              <!-- Subtitle -->\n              <p class="text-xs text-brand-muted font-medium mb-3 px-0.5">\n                Fill in the details below to predict the level of food waste.\n              </p>\n\n              <!-- FORM INPUT ROWS -->\n              <div class="space-y-2.5">\n                \n                <!-- 1. Meal Type -->\n                <div class="app-input-row">\n                  <div class="icon-bubble">\n                    <i data-lucide="utensils" class="w-4 h-4"></i>\n                  </div>\n                  <div class="flex-1 pr-1">\n                    <label for="formMealType" class="block text-[10px] font-bold text-brand-muted uppercase tracking-wider">Meal Type</label>\n                    <div class="relative">\n                      <select id="formMealType" class="w-full bg-transparent text-sm font-bold text-brand-dark focus:outline-none appearance-none cursor-pointer">\n                        <option value="Breakfast">Breakfast</option>\n                        <option value="Lunch" selected>Lunch</option>\n                        <option value="Dinner">Dinner</option>\n                        <option value="Buffet / Event">Buffet / Event</option>\n                      </select>\n                      <i data-lucide="chevron-down" class="w-4 h-4 text-brand-muted absolute right-0 top-1/2 -translate-y-1/2 pointer-events-none"></i>\n                    </div>\n                  </div>\n                </div>\n\n                <!-- 2. Food Prepared (kg) -->\n                <div class="app-input-row">\n                  <div class="icon-bubble">\n                    <i data-lucide="soup" class="w-4 h-4"></i>\n                  </div>\n                  <div class="flex-1">\n                    <label for="formFoodPrepared" class="block text-[10px] font-bold text-brand-muted uppercase tracking-wider">Food Prepared (kg)</label>\n                    <div class="flex items-center justify-between">\n                      <input type="number" id="formFoodPrepared" value="50" min="1" max="1000" class="w-full bg-transparent text-sm font-bold text-brand-dark focus:outline-none" />\n                      <span class="text-xs font-semibold text-brand-muted pr-1">kg</span>\n                    </div>\n                  </div>\n                </div>\n\n                <!-- 3. Meals Served (people) -->\n                <div class="app-input-row">\n                  <div class="icon-bubble">\n                    <i data-lucide="users" class="w-4 h-4"></i>\n                  </div>\n                  <div class="flex-1">\n                    <label for="formMealsServed" class="block text-[10px] font-bold text-brand-muted uppercase tracking-wider">Meals Served</label>\n                    <div class="flex items-center justify-between">\n                      <input type="number" id="formMealsServed" value="45" min="1" max="1000" class="w-full bg-transparent text-sm font-bold text-brand-dark focus:outline-none" />\n                      <span class="text-xs font-semibold text-brand-muted pr-1">people</span>\n                    </div>\n                  </div>\n                </div>\n\n                <!-- 4. Day of the Week -->\n                <div class="app-input-row">\n                  <div class="icon-bubble">\n                    <i data-lucide="calendar" class="w-4 h-4"></i>\n                  </div>\n                  <div class="flex-1 pr-1">\n                    <label for="formDayOfWeek" class="block text-[10px] font-bold text-brand-muted uppercase tracking-wider">Day of the Week</label>\n                    <div class="relative">\n                      <select id="formDayOfWeek" class="w-full bg-transparent text-sm font-bold text-brand-dark focus:outline-none appearance-none cursor-pointer">\n                        <option value="Monday" selected>Monday</option>\n                        <option value="Tuesday">Tuesday</option>\n                        <option value="Wednesday">Wednesday</option>\n                        <option value="Thursday">Thursday</option>\n                        <option value="Friday">Friday</option>\n                        <option value="Saturday">Saturday</option>\n                        <option value="Sunday">Sunday</option>\n                      </select>\n                      <i data-lucide="chevron-down" class="w-4 h-4 text-brand-muted absolute right-0 top-1/2 -translate-y-1/2 pointer-events-none"></i>\n                    </div>\n                  </div>\n                </div>\n\n                <!-- 5. Previous Waste (kg) -->\n                <div class="app-input-row">\n                  <div class="icon-bubble">\n                    <i data-lucide="trash-2" class="w-4 h-4"></i>\n                  </div>\n                  <div class="flex-1">\n                    <label for="formPrevWaste" class="block text-[10px] font-bold text-brand-muted uppercase tracking-wider">Previous Waste (kg)</label>\n                    <div class="flex items-center justify-between">\n                      <input type="number" id="formPrevWaste" value="5" min="0" max="500" step="0.5" class="w-full bg-transparent text-sm font-bold text-brand-dark focus:outline-none" />\n                      <span class="text-xs font-semibold text-brand-muted pr-1">kg</span>\n                    </div>\n                  </div>\n                </div>\n\n              </div>\n            </div>\n\n            <!-- Bottom CTA Button -->\n            <div class="z-10 mt-2">\n              <button id="btnSubmitPredict" onclick="executePrediction()" class="w-full btn-primary py-3.5 px-6 rounded-full font-bold text-sm flex items-center justify-center gap-2 shadow-pill">\n                <i data-lucide="sparkles" class="w-4 h-4 text-emerald-300"></i>\n                <span id="txtSubmitPredict">Predict Waste</span>\n              </button>\n              <div class="w-28 h-1 bg-slate-300 rounded-full mx-auto mt-3 opacity-60"></div>\n            </div>\n          </section>\n\n\n          <!-- ============================================================ -->\n          <!-- SCREEN 3: PREDICTION RESULT -->\n          <!-- ============================================================ -->\n          <section id="scr3" class="app-screen screen-next p-5 pt-0 pb-4 justify-between">\n            <div>\n              <!-- Header Bar -->\n              <div class="flex items-center justify-between mb-1">\n                <button onclick="navTo(2)" class="w-8 h-8 rounded-full bg-white border border-brand-border text-brand-dark flex items-center justify-center hover:bg-brand-mint transition active:scale-95 shadow-xs">\n                  <i data-lucide="arrow-left" class="w-4 h-4"></i>\n                </button>\n                <h2 class="text-sm font-bold text-brand-dark tracking-tight">Prediction Result</h2>\n                <div class="w-8"></div>\n              </div>\n\n              <!-- Mini Bowl Artwork Header -->\n              <div class="flex justify-center my-1">\n                <div class="w-28 h-20 relative">\n                  <svg viewBox="0 0 200 140" class="w-full h-full drop-shadow-sm">\n                    <path d="M30 65 C 10 45, 35 25, 60 30 C 50 50, 45 60, 30 65 Z" fill="#5ec28b" />\n                    <path d="M170 65 C 190 45, 165 25, 140 30 C 150 50, 155 60, 170 65 Z" fill="#80d5a3" />\n                    <path d="M50 85 C 45 60, 75 40, 100 42 C 125 40, 155 60, 150 85 Z" fill="#36835a" />\n                    <circle cx="125" cy="58" r="8" fill="#e74c3c" />\n                    <ellipse cx="85" cy="65" rx="9" ry="7" fill="#ffffff" />\n                    <circle cx="85" cy="65" r="3.5" fill="#f39c12" />\n                    <circle cx="105" cy="72" r="7" fill="#d4a373" />\n                    <circle cx="130" cy="75" r="7" fill="#faedcd" />\n                    <path d="M40 80 C 45 118, 70 128, 100 128 C 130 128, 155 118, 160 80 Z" fill="#1c4a35" />\n                    <ellipse cx="100" cy="80" rx="60" ry="6" fill="#2d6a4f" />\n                  </svg>\n                </div>\n              </div>\n\n              <!-- DYNAMIC RESULT CARD -->\n              <div id="cardResult" class="bg-waste-cream border border-waste-creamBorder rounded-3xl p-4 text-center mb-2.5 shadow-subtle transition-all duration-300">\n                <!-- Circular Icon Badge -->\n                <div id="badgeResultIcon" class="w-9 h-9 rounded-full bg-waste-amberIcon text-white flex items-center justify-center mx-auto mb-1.5 shadow-xs">\n                  <i id="iconResult" data-lucide="bar-chart-2" class="w-4 h-4"></i>\n                </div>\n\n                <!-- Main Level Headline -->\n                <h3 id="txtResultTitle" class="text-xl font-extrabold text-waste-amber tracking-tight">\n                  Medium Waste\n                </h3>\n\n                <!-- Level Description -->\n                <p class="text-xs font-semibold text-stone-700 leading-snug mt-1">\n                  The predicted food waste level<br />for this meal is <span id="txtResultLevel" class="text-waste-amber font-bold">Medium</span>.\n                </p>\n              </div>\n\n              <!-- SUMMARY CARD -->\n              <div class="bg-white rounded-2xl p-3.5 border border-brand-border shadow-subtle mb-2">\n                <h4 class="text-xs font-bold text-brand-dark mb-2 pb-1.5 border-b border-brand-light flex items-center justify-between">\n                  <span>Summary</span>\n                  <span id="tagEfficiency" class="text-[10px] font-semibold px-2 py-0.5 bg-brand-light text-brand-dark rounded-full">Standard</span>\n                </h4>\n                <div class="space-y-1.5 text-xs text-brand-dark">\n                  <div class="flex items-center justify-between">\n                    <span class="text-brand-muted font-medium">Meal Type</span>\n                    <span id="resMealType" class="font-bold">Lunch</span>\n                  </div>\n                  <div class="flex items-center justify-between">\n                    <span class="text-brand-muted font-medium">Food Prepared</span>\n                    <span id="resFoodPrepared" class="font-bold">50 kg</span>\n                  </div>\n                  <div class="flex items-center justify-between">\n                    <span class="text-brand-muted font-medium">Meals Served</span>\n                    <span id="resMealsServed" class="font-bold">45</span>\n                  </div>\n                  <div class="flex items-center justify-between">\n                    <span class="text-brand-muted font-medium">Day</span>\n                    <span id="resDay" class="font-bold">Monday</span>\n                  </div>\n                  <div class="flex items-center justify-between">\n                    <span class="text-brand-muted font-medium">Previous Waste</span>\n                    <span id="resPrevWaste" class="font-bold">5 kg</span>\n                  </div>\n                </div>\n              </div>\n            </div>\n\n            <!-- Bottom CTA Button & Footer Slogan -->\n            <div class="z-10">\n              <button onclick="navTo(2)" class="w-full btn-primary py-3.5 px-6 rounded-full font-bold text-sm flex items-center justify-center gap-2 shadow-pill mb-2">\n                <i data-lucide="rotate-cw" class="w-4 h-4"></i>\n                <span>Predict Again</span>\n              </button>\n\n              <div class="flex items-center justify-center gap-1.5 text-center text-[10.5px] font-semibold text-brand-muted">\n                <i data-lucide="leaf" class="w-3.5 h-3.5 text-brand-emerald"></i>\n                <span>Less food waste today, a healthier tomorrow.</span>\n              </div>\n\n              <div class="w-28 h-1 bg-slate-300 rounded-full mx-auto mt-2.5 opacity-60"></div>\n            </div>\n          </section>\n\n        </div>\n      </div>\n    </div>\n  </main>\n\n\n  <!-- ========================================================================= -->\n  <!-- 2. TRIPLE SCREEN SHOWCASE (Side-by-side view exactly matching screenshot) -->\n  <!-- ========================================================================= -->\n  <section id="viewShowcaseContainer" class="hidden w-full max-w-6xl py-2 overflow-x-auto">\n    <div class="flex flex-col lg:flex-row items-center justify-center gap-6 min-w-max mx-auto px-4">\n      \n      <!-- 1. Screen 1 (Welcome) -->\n      <div class="flex flex-col items-center">\n        <div class="iphone-frame scale-[0.88] origin-top">\n          <div class="iphone-screen p-5 pt-1 pb-5 flex flex-col justify-between">\n            <div class="h-9 flex items-center justify-between px-3 text-[11px] font-bold text-brand-dark">\n              <span>9:41</span>\n              <div class="dynamic-island"></div>\n              <span>●●●</span>\n            </div>\n            \n            <div class="flex flex-col items-center text-center mt-1">\n              <div class="w-12 h-12 bg-brand-dark rounded-2xl flex items-center justify-center text-white mb-2 shadow-md">\n                <i data-lucide="leaf" class="w-6 h-6 text-emerald-300"></i>\n              </div>\n              <h1 class="text-2xl font-black text-brand-dark leading-tight">Food Waste<br/><span class="text-brand-emerald text-xl">Level Prediction</span></h1>\n              <p class="text-brand-muted text-xs font-semibold mt-1">Smarter decisions for less food waste.</p>\n            </div>\n\n            <!-- Illustration -->\n            <div class="my-auto flex justify-center">\n              <div class="w-56 h-44">\n                <svg viewBox="0 0 340 280" class="w-full h-full drop-shadow-md">\n                  <path d="M60 110 C 35 75, 75 35, 120 45 C 105 85, 95 105, 60 110 Z" fill="#5ec28b" />\n                  <path d="M280 100 C 315 65, 290 25, 245 35 C 255 75, 265 95, 280 100 Z" fill="#80d5a3" />\n                  <path d="M65 155 C 55 110, 110 65, 170 70 C 230 65, 285 110, 275 155 Z" fill="#36835a" />\n                  <circle cx="230" cy="105" r="18" fill="#e74c3c" />\n                  <circle cx="115" cy="148" r="22" fill="#e74c3c" />\n                  <ellipse cx="185" cy="118" rx="20" ry="15" fill="#ffffff" />\n                  <circle cx="185" cy="118" r="9" fill="#f39c12" />\n                  <circle cx="145" cy="125" r="16" fill="#d4a373" />\n                  <path d="M45 145 C 50 220, 95 245, 170 245 C 245 245, 290 220, 295 145 C 295 136, 45 136, 45 145 Z" fill="#1c4a35" />\n                  <ellipse cx="170" cy="143" rx="125" ry="11" fill="#2d6a4f" />\n                </svg>\n              </div>\n            </div>\n\n            <!-- 3 Badges -->\n            <div class="grid grid-cols-3 gap-1.5 mb-3 text-center">\n              <div class="p-2 rounded-xl bg-white border border-brand-border text-[9px] font-bold text-brand-dark">\n                <i data-lucide="leaf" class="w-3.5 h-3.5 mx-auto mb-1 text-brand-emerald"></i>Reduce<br/>Food Waste\n              </div>\n              <div class="p-2 rounded-xl bg-white border border-brand-border text-[9px] font-bold text-brand-dark">\n                <i data-lucide="globe" class="w-3.5 h-3.5 mx-auto mb-1 text-brand-emerald"></i>Save<br/>Resources\n              </div>\n              <div class="p-2 rounded-xl bg-white border border-brand-border text-[9px] font-bold text-brand-dark">\n                <i data-lucide="sprout" class="w-3.5 h-3.5 mx-auto mb-1 text-brand-emerald"></i>Build a<br/>Greener Future\n              </div>\n            </div>\n\n            <button onclick="switchMainView(\'interactive\'); navTo(2);" class="w-full btn-primary py-3 rounded-full font-bold text-xs flex items-center justify-center gap-1.5 shadow">\n              <span>Predict Waste</span>\n              <i data-lucide="arrow-right" class="w-3.5 h-3.5"></i>\n            </button>\n          </div>\n        </div>\n      </div>\n\n      <!-- 2. Screen 2 (Meal Details) -->\n      <div class="flex flex-col items-center">\n        <div class="iphone-frame scale-[0.88] origin-top">\n          <div class="iphone-screen p-5 pt-1 pb-5 flex flex-col justify-between">\n            <div class="h-9 flex items-center justify-between px-3 text-[11px] font-bold text-brand-dark">\n              <span>9:41</span>\n              <div class="dynamic-island"></div>\n              <span>●●●</span>\n            </div>\n\n            <div>\n              <div class="flex items-center justify-between mb-2">\n                <i data-lucide="arrow-left" class="w-4 h-4 text-brand-dark"></i>\n                <h2 class="text-sm font-bold text-brand-dark">Enter Meal Details</h2>\n                <div class="w-4"></div>\n              </div>\n              <p class="text-xs text-brand-muted mb-3">Fill in the details below to predict the level of food waste.</p>\n\n              <div class="space-y-2.5 text-xs">\n                <div class="app-input-row py-2">\n                  <div class="icon-bubble w-8 h-8"><i data-lucide="utensils" class="w-4 h-4"></i></div>\n                  <div class="flex-1">\n                    <div class="text-[10px] font-bold text-brand-muted uppercase">Meal Type</div>\n                    <div class="font-bold text-brand-dark flex justify-between">Lunch <i data-lucide="chevron-down" class="w-3.5 h-3.5 opacity-60"></i></div>\n                  </div>\n                </div>\n                <div class="app-input-row py-2">\n                  <div class="icon-bubble w-8 h-8"><i data-lucide="soup" class="w-4 h-4"></i></div>\n                  <div class="flex-1">\n                    <div class="text-[10px] font-bold text-brand-muted uppercase">Food Prepared (kg)</div>\n                    <div class="font-bold text-brand-dark flex justify-between">50 <span class="font-normal text-stone-500">kg</span></div>\n                  </div>\n                </div>\n                <div class="app-input-row py-2">\n                  <div class="icon-bubble w-8 h-8"><i data-lucide="users" class="w-4 h-4"></i></div>\n                  <div class="flex-1">\n                    <div class="text-[10px] font-bold text-brand-muted uppercase">Meals Served</div>\n                    <div class="font-bold text-brand-dark flex justify-between">45 <span class="font-normal text-stone-500">people</span></div>\n                  </div>\n                </div>\n                <div class="app-input-row py-2">\n                  <div class="icon-bubble w-8 h-8"><i data-lucide="calendar" class="w-4 h-4"></i></div>\n                  <div class="flex-1">\n                    <div class="text-[10px] font-bold text-brand-muted uppercase">Day of the Week</div>\n                    <div class="font-bold text-brand-dark flex justify-between">Monday <i data-lucide="chevron-down" class="w-3.5 h-3.5 opacity-60"></i></div>\n                  </div>\n                </div>\n                <div class="app-input-row py-2">\n                  <div class="icon-bubble w-8 h-8"><i data-lucide="trash-2" class="w-4 h-4"></i></div>\n                  <div class="flex-1">\n                    <div class="text-[10px] font-bold text-brand-muted uppercase">Previous Waste (kg)</div>\n                    <div class="font-bold text-brand-dark flex justify-between">5 <span class="font-normal text-stone-500">kg</span></div>\n                  </div>\n                </div>\n              </div>\n            </div>\n\n            <button onclick="switchMainView(\'interactive\'); navTo(3);" class="w-full btn-primary py-3 rounded-full font-bold text-xs flex items-center justify-center gap-1.5 shadow mt-2">\n              <i data-lucide="sparkles" class="w-3.5 h-3.5 text-emerald-300"></i>\n              <span>Predict Waste</span>\n            </button>\n          </div>\n        </div>\n      </div>\n\n      <!-- 3. Screen 3 (Prediction Result) -->\n      <div class="flex flex-col items-center">\n        <div class="iphone-frame scale-[0.88] origin-top">\n          <div class="iphone-screen p-5 pt-1 pb-4 flex flex-col justify-between">\n            <div class="h-9 flex items-center justify-between px-3 text-[11px] font-bold text-brand-dark">\n              <span>9:41</span>\n              <div class="dynamic-island"></div>\n              <span>●●●</span>\n            </div>\n\n            <div>\n              <div class="flex items-center justify-between mb-1">\n                <i data-lucide="arrow-left" class="w-4 h-4 text-brand-dark"></i>\n                <h2 class="text-sm font-bold text-brand-dark">Prediction Result</h2>\n                <div class="w-4"></div>\n              </div>\n\n              <!-- Mini Bowl Artwork -->\n              <div class="flex justify-center my-0.5">\n                <div class="w-24 h-16">\n                  <svg viewBox="0 0 200 135" class="w-full h-full drop-shadow-sm">\n                    <path d="M50 85 C 45 60, 75 40, 100 42 C 125 40, 155 60, 150 85 Z" fill="#36835a" />\n                    <circle cx="125" cy="58" r="8" fill="#e74c3c" />\n                    <ellipse cx="85" cy="65" rx="9" ry="7" fill="#ffffff" />\n                    <circle cx="85" cy="65" r="3.5" fill="#f39c12" />\n                    <path d="M40 80 C 45 118, 70 128, 100 128 C 130 128, 155 118, 160 80 Z" fill="#1c4a35" />\n                    <ellipse cx="100" cy="80" rx="60" ry="6" fill="#2d6a4f" />\n                  </svg>\n                </div>\n              </div>\n\n              <!-- Result Card -->\n              <div class="bg-waste-cream border border-waste-creamBorder rounded-2xl p-3.5 text-center mb-2 shadow-sm">\n                <div class="w-8 h-8 rounded-full bg-waste-amberIcon text-white flex items-center justify-center mx-auto mb-1">\n                  <i data-lucide="bar-chart-2" class="w-4 h-4"></i>\n                </div>\n                <h3 class="text-base font-extrabold text-waste-amber">Medium Waste</h3>\n                <p class="text-xs font-semibold text-stone-700 mt-0.5">\n                  The predicted food waste level for this meal is <span class="text-waste-amber font-bold">Medium</span>.\n                </p>\n              </div>\n\n              <!-- Summary -->\n              <div class="bg-white rounded-xl p-3 border border-brand-border text-xs space-y-1">\n                <div class="font-bold text-brand-dark pb-1 border-b border-brand-light">Summary</div>\n                <div class="flex justify-between text-brand-dark"><span>Meal Type</span><span class="font-bold">Lunch</span></div>\n                <div class="flex justify-between text-brand-dark"><span>Food Prepared</span><span class="font-bold">50 kg</span></div>\n                <div class="flex justify-between text-brand-dark"><span>Meals Served</span><span class="font-bold">45</span></div>\n                <div class="flex justify-between text-brand-dark"><span>Day</span><span class="font-bold">Monday</span></div>\n                <div class="flex justify-between text-brand-dark"><span>Previous Waste</span><span class="font-bold">5 kg</span></div>\n              </div>\n            </div>\n\n            <div>\n              <button onclick="switchMainView(\'interactive\'); navTo(2);" class="w-full btn-primary py-3 rounded-full font-bold text-xs flex items-center justify-center gap-1.5 shadow mb-2">\n                <i data-lucide="rotate-cw" class="w-3.5 h-3.5"></i>\n                <span>Predict Again</span>\n              </button>\n              <div class="text-[10px] text-center text-brand-muted flex items-center justify-center gap-1 font-semibold">\n                <i data-lucide="leaf" class="w-3 h-3 text-brand-emerald"></i>\n                <span>Less food waste today, a healthier tomorrow.</span>\n              </div>\n            </div>\n\n          </div>\n        </div>\n      </div>\n\n    </div>\n  </section>\n\n  <!-- ========================================================================= -->\n  <!-- JAVASCRIPT LOGIC -->\n  <!-- ========================================================================= -->\n  <script>\n    // Initialize Lucide Icons\n    lucide.createIcons();\n\n    // Real-time Clock\n    function refreshClock() {\n      const now = new Date();\n      const hh = now.getHours().toString().padStart(2, \'0\');\n      const mm = now.getMinutes().toString().padStart(2, \'0\');\n      const clockElem = document.getElementById(\'liveClock\');\n      if (clockElem) clockElem.textContent = `${hh}:${mm}`;\n    }\n    setInterval(refreshClock, 10000);\n    refreshClock();\n\n    // App Screen Navigation\n    function navTo(screenNum) {\n      const s1 = document.getElementById(\'scr1\');\n      const s2 = document.getElementById(\'scr2\');\n      const s3 = document.getElementById(\'scr3\');\n\n      [s1, s2, s3].forEach((screenEl, index) => {\n        const num = index + 1;\n        screenEl.className = \'app-screen\';\n        if (num === screenNum) {\n          screenEl.classList.add(\'screen-active\');\n        } else if (num < screenNum) {\n          screenEl.classList.add(\'screen-prev\');\n        } else {\n          screenEl.classList.add(\'screen-next\');\n        }\n      });\n\n      const activeEl = document.getElementById(`scr${screenNum}`);\n      if (activeEl) activeEl.scrollTop = 0;\n    }\n\n    // Prediction Execution with Backend FastAPI Connection & Smooth Loader\n    async function executePrediction() {\n      const btn = document.getElementById(\'btnSubmitPredict\');\n      const txt = document.getElementById(\'txtSubmitPredict\');\n\n      btn.disabled = true;\n      txt.textContent = \'Calculating waste...\';\n      btn.classList.add(\'opacity-80\');\n\n      const mealType = document.getElementById(\'formMealType\').value;\n      const foodPrepared = parseFloat(document.getElementById(\'formFoodPrepared\').value) || 0;\n      const mealsServed = parseFloat(document.getElementById(\'formMealsServed\').value) || 1;\n      const dayOfWeek = document.getElementById(\'formDayOfWeek\').value;\n      const prevWaste = parseFloat(document.getElementById(\'formPrevWaste\').value) || 0;\n\n      const payload = {\n        meal_type: mealType,\n        food_prepared_kg: foodPrepared,\n        meals_served: mealsServed,\n        day_of_week: dayOfWeek,\n        prev_waste_kg: prevWaste\n      };\n\n      try {\n        const apiUrl = window.location.origin && window.location.origin.startsWith(\'http\') ? \'/predict\' : \'http://127.0.0.1:8000/predict\';\n        const response = await fetch(apiUrl, {\n          method: \'POST\',\n          headers: { \'Content-Type\': \'application/json\' },\n          body: JSON.stringify(payload)\n        });\n\n        if (!response.ok) {\n          throw new Error(`Server returned status ${response.status}`);\n        }\n\n        const data = await response.json();\n        renderPredictionResult(data.predicted_category, data.efficiency_tag, true, mealType, foodPrepared, mealsServed, dayOfWeek, prevWaste);\n      } catch (err) {\n        console.warn(\'Backend API unavailable, falling back to local ML heuristic:\', err);\n        computeWasteModel(true);\n      } finally {\n        navTo(3);\n        btn.disabled = false;\n        txt.textContent = \'Predict Waste\';\n        btn.classList.remove(\'opacity-80\');\n      }\n    }\n\n    // Render Prediction Output to Screen 3\n    function renderPredictionResult(category, efficiencyTag, triggerConfetti, mealType, foodPrepared, mealsServed, dayOfWeek, prevWaste) {\n      let level = \'Medium\';\n      let titleColorClass = \'text-waste-amber\';\n      let cardBg = \'#fdf7e3\';\n      let cardBorder = \'#faecc5\';\n      let iconBadgeBg = \'#d97706\';\n      let iconName = \'bar-chart-2\';\n      let efficiencyText = efficiencyTag || \'Standard Margin\';\n\n      if (category.toLowerCase().includes(\'low\')) {\n        level = \'Low\';\n        titleColorClass = \'text-waste-greenText\';\n        cardBg = \'#eef8f2\';\n        cardBorder = \'#d0ebd8\';\n        iconBadgeBg = \'#2e8b57\';\n        iconName = \'check-circle-2\';\n        if (!efficiencyTag) efficiencyText = \'High Efficiency 🌱\';\n\n        if (triggerConfetti && typeof confetti === \'function\') {\n          confetti({\n            particleCount: 45,\n            spread: 55,\n            origin: { y: 0.62 },\n            colors: [\'#2e8b57\', \'#5ec28b\', \'#f39c12\', \'#80d5a3\']\n          });\n        }\n      } else if (category.toLowerCase().includes(\'high\')) {\n        level = \'High\';\n        titleColorClass = \'text-waste-redText\';\n        cardBg = \'#fef1f0\';\n        cardBorder = \'#fcd3cf\';\n        iconBadgeBg = \'#e04f3d\';\n        iconName = \'alert-triangle\';\n        if (!efficiencyTag) efficiencyText = \'Surplus Alert ⚠️\';\n      }\n\n      // Update Screen 3 Visual Elements\n      const card = document.getElementById(\'cardResult\');\n      const title = document.getElementById(\'txtResultTitle\');\n      const levelTxt = document.getElementById(\'txtResultLevel\');\n      const iconBadge = document.getElementById(\'badgeResultIcon\');\n      const icon = document.getElementById(\'iconResult\');\n      const tag = document.getElementById(\'tagEfficiency\');\n\n      if (card) {\n        card.style.backgroundColor = cardBg;\n        card.style.borderColor = cardBorder;\n      }\n      if (title) {\n        title.className = `text-xl font-extrabold tracking-tight ${titleColorClass}`;\n        title.textContent = `${level} Waste`;\n      }\n      if (levelTxt) {\n        levelTxt.className = `font-bold ${titleColorClass}`;\n        levelTxt.textContent = level;\n      }\n      if (iconBadge) {\n        iconBadge.style.backgroundColor = iconBadgeBg;\n      }\n      if (tag) {\n        tag.textContent = efficiencyText;\n      }\n      if (icon) {\n        icon.setAttribute(\'data-lucide\', iconName);\n      }\n\n      // Summary Card Text\n      const elMealType = document.getElementById(\'resMealType\');\n      const elFoodPrep = document.getElementById(\'resFoodPrepared\');\n      const elMealsServ = document.getElementById(\'resMealsServed\');\n      const elDay = document.getElementById(\'resDay\');\n      const elPrevWaste = document.getElementById(\'resPrevWaste\');\n\n      if (elMealType) elMealType.textContent = mealType;\n      if (elFoodPrep) elFoodPrep.textContent = `${foodPrepared} kg`;\n      if (elMealsServ) elMealsServ.textContent = mealsServed.toString();\n      if (elDay) elDay.textContent = dayOfWeek;\n      if (elPrevWaste) elPrevWaste.textContent = `${prevWaste} kg`;\n\n      if (window.lucide && typeof lucide.createIcons === \'function\') {\n        lucide.createIcons();\n      }\n    }\n\n    // Dynamic Food Waste Prediction Fallback Engine\n    function computeWasteModel(triggerConfetti = false) {\n      const mealType = document.getElementById(\'formMealType\').value;\n      const foodPrepared = parseFloat(document.getElementById(\'formFoodPrepared\').value) || 0;\n      const mealsServed = parseFloat(document.getElementById(\'formMealsServed\').value) || 1;\n      const dayOfWeek = document.getElementById(\'formDayOfWeek\').value;\n      const prevWaste = parseFloat(document.getElementById(\'formPrevWaste\').value) || 0;\n\n      // Realistic consumption benchmark (kg per person)\n      let portion = 0.65;\n      if (mealType === \'Breakfast\') portion = 0.40;\n      if (mealType === \'Dinner\') portion = 0.75;\n      if (mealType.includes(\'Buffet\')) portion = 0.85;\n\n      const expectedConsumed = mealsServed * portion;\n      const excessKg = foodPrepared - expectedConsumed;\n      const excessRatio = foodPrepared > 0 ? (excessKg / foodPrepared) : 0;\n      const historyRatio = foodPrepared > 0 ? (prevWaste / foodPrepared) : 0;\n\n      // Composite prediction index\n      const wasteIndex = (excessRatio * 0.7) + (historyRatio * 0.3);\n\n      let category = \'Medium Waste\';\n      let tag = \'Standard Margin\';\n\n      if (wasteIndex < 0.08 && excessRatio < 0.12) {\n        category = \'Low Waste\';\n        tag = \'High Efficiency 🌱\';\n      } else if (wasteIndex >= 0.22 || excessRatio > 0.28) {\n        category = \'High Waste\';\n        tag = \'Surplus Alert ⚠️\';\n      }\n\n      renderPredictionResult(category, tag, triggerConfetti, mealType, foodPrepared, mealsServed, dayOfWeek, prevWaste);\n    }\n\n    // Preset Scenarios\n    function setScenario(type) {\n      if (type === \'medium\') {\n        document.getElementById(\'formMealType\').value = \'Lunch\';\n        document.getElementById(\'formFoodPrepared\').value = \'50\';\n        document.getElementById(\'formMealsServed\').value = \'45\';\n        document.getElementById(\'formDayOfWeek\').value = \'Monday\';\n        document.getElementById(\'formPrevWaste\').value = \'5\';\n      } else if (type === \'low\') {\n        document.getElementById(\'formMealType\').value = \'Lunch\';\n        document.getElementById(\'formFoodPrepared\').value = \'32\';\n        document.getElementById(\'formMealsServed\').value = \'48\';\n        document.getElementById(\'formDayOfWeek\').value = \'Wednesday\';\n        document.getElementById(\'formPrevWaste\').value = \'1.5\';\n      } else if (type === \'high\') {\n        document.getElementById(\'formMealType\').value = \'Dinner\';\n        document.getElementById(\'formFoodPrepared\').value = \'85\';\n        document.getElementById(\'formMealsServed\').value = \'40\';\n        document.getElementById(\'formDayOfWeek\').value = \'Saturday\';\n        document.getElementById(\'formPrevWaste\').value = \'18\';\n      }\n      computeWasteModel(false);\n      navTo(2);\n    }\n\n    // Main View Switching\n    function switchMainView(mode) {\n      const iContainer = document.getElementById(\'viewInteractiveContainer\');\n      const sContainer = document.getElementById(\'viewShowcaseContainer\');\n      const tabI = document.getElementById(\'tabInteractive\');\n      const tabS = document.getElementById(\'tabShowcase\');\n\n      if (mode === \'interactive\') {\n        iContainer.classList.remove(\'hidden\');\n        sContainer.classList.add(\'hidden\');\n        tabI.className = \'flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition-all bg-emerald-700 text-white shadow-sm\';\n        tabS.className = \'flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition-all text-slate-300 hover:text-white\';\n      } else {\n        iContainer.classList.add(\'hidden\');\n        sContainer.classList.remove(\'hidden\');\n        tabS.className = \'flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition-all bg-emerald-700 text-white shadow-sm\';\n        tabI.className = \'flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition-all text-slate-300 hover:text-white\';\n      }\n      lucide.createIcons();\n    }\n\n    // Initialize Default State\n    computeWasteModel(false);\n    navTo(1);\n  </script>\n</body>\n</html>\n'

# =====================================================================
# 6. CLI DIRECT EXECUTION WITH AUTO-BROWSER LAUNCH
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import time

    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()

    print("=" * 65)
    print("  * Food Waste Level Prediction System")
    print("  * Interactive Web UI:     http://127.0.0.1:8000")
    print("  * API Documentation:     http://127.0.0.1:8000/docs")
    print("=" * 65)
    uvicorn.run(app, host="127.0.0.1", port=8000)
