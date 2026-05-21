# AI-Based-Car-Price-Prediction-System
 Developed a machine learning system to predict used car prices from historical vehicle data.

# 🚗 AI Based Car Price Prediction System

An end-to-end machine learning project designed to predict used car prices using historical vehicle data. The system applies data preprocessing, feature engineering, model comparison, and cross-validation to generate reliable price predictions and is deployed through an interactive Streamlit web application.

---

## 📌 Project Overview

Used car pricing is influenced by multiple factors and often follows non-linear market patterns. This project aims to build a machine learning-based valuation system that predicts resale prices using historical vehicle data.

The system performs:

- Data cleaning and preprocessing
- Feature engineering and encoding
- Multi-model comparison
- Cross-validation and performance evaluation
- Deployment using Streamlit

---

## 🚀 Features

- Predicts used car resale prices
- Handles noisy and real-world vehicle data
- Implements feature engineering and categorical encoding
- Compares multiple machine learning models
- Uses 5-Fold Cross Validation for robust evaluation
- Interactive Streamlit web interface
- Model analytics and feature importance visualization

---

## 🛠️ Tech Stack

**Programming Language**
- Python

**Libraries & Frameworks**
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Joblib
- Streamlit

---

## 📂 Dataset

Dataset used:

**Car Details v3 Dataset**

The dataset contains historical vehicle information and selling prices used for supervised machine learning.

---

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Cleaned numerical fields using Regex
- Handled missing values
- Converted raw inputs into usable numerical features
- Generated derived features such as car age

### 2. Feature Engineering
- Extracted brand and model information
- Applied encoding techniques for categorical handling
- Prepared model-ready input features

### 3. Model Training & Comparison
The following regression models were trained and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

### 4. Model Evaluation
Models were evaluated using:

- 5-Fold Cross Validation
- R² Score
- Mean Absolute Error (MAE)

The final model was selected based on cross-validation performance.

### 5. Deployment
The selected model was deployed using **Streamlit** to provide real-time car price prediction.

---

## 📊 Application Modules

### 🔹 Price Predictor
Allows users to enter vehicle details and estimate resale value.

### 🔹 Model Analytics
Displays:
- Model comparison results
- Cross-validation performance
- Feature importance

### 🔹 About Project
Provides project documentation and technical overview.

---

## 📁 Project Structure

```bash
AI-Based-Car-Price-Prediction-System/
│
├── train_model.py
├── app.py
├── Car details v3.csv
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1. Clone Repository

```bash
git clone <repository-link>
cd AI-Based-Car-Price-Prediction-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

```bash
python train_model.py
```

### 4. Run Streamlit App

```bash
streamlit run app.py
```

---

## 📈 Future Improvements

Possible enhancements:

- Hyperparameter tuning
- Geographical pricing factors
- Image-based vehicle condition assessment
- Additional explainability techniques

---

## 👨‍💻 Author

**Yuvanesh S**  
