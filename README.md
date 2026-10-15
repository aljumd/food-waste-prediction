# Food Waste Level Prediction — Machine Learning System

A machine learning application and REST API that predicts operational food waste levels (**Low Waste**, **Medium Waste**, **High Waste**) for commercial catering, cafeteria dining, and restaurant buffet services based on meal planning parameters and portion metrics.

---

## 1. Project Overview & Problem Statement

Food waste in commercial kitchens, university cafeterias, and hotel buffets leads to substantial financial loss and environmental impact. The primary causes of commercial food waste are:
1. **Demand Misestimation**: Preparing significantly more food than attendees consume.
2. **Inflexible Safety Margins**: Applying uniform buffer percentages across diverse meal types and days without considering appetite benchmarks.
3. **Historical Inefficiencies**: Ignoring previous meal waste trends in current meal planning.

### Objective
To build an end-to-end Machine Learning solution that takes operational meal parameters before a service and accurately classifies the expected waste into actionable categories:
* **Low Waste ($\le 12\%$)**: Lean, well-calibrated preparation with minimal unserved food.
* **Medium Waste ($12\% - 25\%$)**: Standard operational safety margin.
* **High Waste ($> 25\%$)**: Surplus alert requiring immediate portion reduction.

---

## 2. Dataset Description & Methodology

### Dataset Disclosure
* **Dataset File**: `dataset.csv`
* **Size**: 3,000 meal service observations (80/20 train/test split: 2,400 training / 600 test records).
* **Source & Nature**: Statistically simulated benchmark dataset grounded in published catering and food service standards (WRAP Hospitality & FAO Catering Waste Frameworks), generated with fixed seed `random_seed=42`.
* **Portion Appetite Benchmarks**:
  * **Breakfast**: $0.35 - 0.48\text{ kg/person}$ (Continental/light meal)
  * **Lunch**: $0.50 - 0.68\text{ kg/person}$ (Standard midday meal)
  * **Dinner**: $0.65 - 0.85\text{ kg/person}$ (Multi-course dinner)
  * **Buffet / Event**: $0.75 - 1.05\text{ kg/person}$ (High-variety banquet dining)

### Feature Schema

| Column Name | Data Type | Range / Values | Description |
| :--- | :---: | :---: | :--- |
| `meal_type` | Categorical | `Breakfast`, `Lunch`, `Dinner`, `Buffet / Event` | Service category |
| `food_prepared_kg` | Numerical (float) | $5.0 - 500.0\text{ kg}$ | Total food cooked/prepared |
| `meals_served` | Numerical (int) | $10 - 450\text{ diners}$ | Total diners / tickets served |
| `day_of_week` | Categorical | `Monday` – `Sunday` | Day of service |
| `previous_waste_kg` | Numerical (float) | $0.0 - 120.0\text{ kg}$ | Waste from previous meal cycle |
| `actual_waste_kg` | Numerical (float) | $0.15 - 105.0\text{ kg}$ | *Target variable* (continuous waste in kg) |
| `waste_percentage` | Numerical (float) | $2.0\% - 48.0\%$ | $(\text{actual\_waste} / \text{food\_prepared}) \times 100$ |

---

## ⚙️ 3. Preprocessing & Feature Engineering

1. **Domain Feature Engineering (`FoodWasteFeatureEngineer`)**:
   * $\text{prepared\_per\_person} = \frac{\text{food\_prepared\_kg}}{\max(\text{meals\_served}, 1)}$
   * $\text{prev\_waste\_ratio} = \frac{\text{previous\_waste\_kg}}{\max(\text{food\_prepared\_kg}, 0.1)}$
   * $\text{portion\_excess\_ratio} = \frac{\text{prepared\_per\_person} - \text{baseline\_portion}}{\text{baseline\_portion}}$
2. **Column Transformations**:
   * Categorical features (`meal_type`, `day_of_week`) encoded with `OneHotEncoder(handle_unknown='ignore')`.
   * Numerical features scaled with `StandardScaler()`.
3. **Data Leakage Prevention**:
   * Continuous target measurements (`actual_waste_kg`, `waste_percentage`) are excluded from features $X$ during training and inference.
   * Categorization thresholds are calculated strictly on training set quantiles.

---

## 🔬 4. Machine Learning Algorithms & Model Comparison

Three classification algorithms were trained and evaluated on an 80/20 stratified train-test split (`random_state=42`) with 5-fold cross-validation. (Linear Regression and Logistic Regression were excluded per project specifications).

| Model | 5-Fold CV Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score | ROC-AUC (OvR Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  **Random Forest (Selected)** | **86.46%** | **87.67%** | **0.8959** | **0.8670** | **0.8791** | **0.8770** | **0.9464** |
| **Decision Tree Classifier** | 85.62% | 87.67% | 0.9014 | 0.8629 | 0.8782 | 0.8768 | 0.9415 |
| **K-Nearest Neighbors (KNN)** | 83.96% | 83.83% | 0.8651 | 0.8225 | 0.8386 | 0.8384 | 0.9341 |

* **Selected Model**: **Random Forest Classifier** ($150$ estimators, `max_depth=10`, `min_samples_split=4`, `min_samples_leaf=2`) saved to `model.pkl`.

---

##  5. Installation & Execution

### Prerequisites
* Python 3.9+

### Setup Virtual Environment
```powershell
cd food-waste-prediction
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Start the REST API & Web Service
```powershell
python app.py
```
* Interactive API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`

---

##  6. Testing & API Reference

### Endpoints
* `GET /`: API status and route catalog.
* `GET /health`: Health status and model loaded check.
* `GET /model-info`: Active model metadata and thresholds.
* `POST /predict`: Real-time prediction endpoint.

### Sample Prediction Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "meal_type": "Lunch",
       "food_prepared_kg": 32.0,
       "meals_served": 48,
       "day_of_week": "Wednesday",
       "prev_waste_kg": 1.5
     }'
```

### Sample Prediction Response
```json
{
  "predicted_category": "Low Waste",
  "confidence": 0.82,
  "probabilities": {
    "High Waste": 0.0084,
    "Low Waste": 0.8201,
    "Medium Waste": 0.1715
  },
  "recommendation_message": "The predicted food waste level for this lunch is Low. Food preparation is well-calibrated (0.67 kg/person).",
  "efficiency_tag": "High Efficiency 🌱",
  "waste_reduction_tips": [
    "Maintain current batch cooking portions.",
    "Log unserved leftovers for surplus donation programs.",
    "Keep tracking attendance patterns across weekdays."
  ],
  "summary": {
    "meal_type": "Lunch",
    "food_prepared": "32.0 kg",
    "meals_served": "48",
    "day": "Wednesday",
    "previous_waste": "1.5 kg"
  }
}
```

---

## 7. Limitations & Recommendations for Real-World Deployment

1. **Synthetic Nature of Training Data**: The dataset was statistically generated using catering benchmark distributions. While realistic, it does not capture site-specific menu variations (e.g., salad bar vs. pasta bar).
2. **Ingredient-Level Granularity**: The model evaluates total kilograms rather than individual dish categories (proteins, grains, vegetables).
3. **Recommendation for Production**: Cafeterias deploying this system should log daily weigh-scale smart bin measurements for 30–60 days to fine-tune the pipeline on genuine kitchen telemetry.
