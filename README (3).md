# Forest Fire Burned Area Prediction using Machine Learning

This project predicts the **burned area (in hectares) caused by forest fires** using a wide range of machine learning regression models.  
It includes **data preprocessing, extensive model comparison (20+ models), visualization, and an interactive prediction interface**.

---

## 🔥 Problem Statement
Forest fires cause severe environmental and economic damage. Predicting the burned area based on weather, fuel moisture, and spatial data can help in **risk assessment and disaster mitigation planning**.

This project uses regression-based machine learning models to estimate the expected burned area of forest fires.

---

## 📊 Dataset
- **Source**: UCI Forest Fires Dataset
- **Records**: 517
- **Features**: 12 input attributes
- **Target Variable**:
  - `area` (burned area in hectares)

To reduce skewness, a **log transformation** is applied:
```
log_area = log(area + 1)
```

---

## ⚙️ Technologies Used
- **Language**: Python
- **Libraries**:
  - pandas, numpy
  - scikit-learn
  - matplotlib
  - ipywidgets
  - (Optional) XGBoost, CatBoost, LightGBM

---

## 🔄 Project Workflow

1. Data loading and inspection  
2. Encoding categorical features (month, day)  
3. Missing value handling using mean imputation  
4. Log transformation of target variable  
5. Feature scaling using StandardScaler  
6. Training **20+ regression models**  
7. Model comparison using Mean Squared Error (MSE)  
8. Visualization using 1 / MSE  
9. Interactive burned-area prediction interface  

---

## 🧠 Models Implemented

### Core Models
- Linear Regression
- KNN Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor

### Advanced & Ensemble Models
- Ridge, Lasso, Elastic Net
- Bayesian Ridge, Huber Regressor
- Extra Trees, AdaBoost, Bagging
- SGD Regressor
- Kernel Ridge
- Gaussian Process
- MLP Regressor

### Optional Boosting Models
- XGBoost
- CatBoost
- LightGBM

---

## 📈 Model Evaluation
- **Metric**: Mean Squared Error (MSE)
- **Visualization Metric**: Accuracy = `1 / MSE`
- Bar chart comparison across all trained models

---

## 🧪 Interactive Prediction
Users can input:
- Spatial coordinates
- Month and day
- Fire weather indices (FFMC, DMC, DC, ISI)
- Temperature, humidity, wind speed, and rainfall

The system predicts burned area using **all trained models simultaneously**.

---

## ▶️ How to Run

1. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib ipywidgets
```

(Optional)
```bash
pip install xgboost catboost lightgbm
```

2. Place dataset in project directory:
```
forestfires.csv
```

3. Run the notebook or script:
```bash
python forest_fire_prediction.py
```

4. Use the interactive widgets to generate predictions.

---

## 🚀 Future Enhancements
- Hyperparameter tuning
- Model persistence (joblib)
- Streamlit or Flask web deployment
- Feature importance visualization
- Time-series fire risk analysis

---

## 👨‍💻 Author
**Tanish Sindwani**  
BTech CSE | Machine Learning & Data Analytics

⭐ Star the repo if you find it useful!
