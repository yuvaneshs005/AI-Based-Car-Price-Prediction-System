import pandas as pd
import numpy as np
import re
import datetime
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load Data
df = pd.read_csv('Car details v3.csv')

# --- DATA PREPROCESSING ---
def clean_numeric(val):
    if pd.isna(val) or val == 'null': return np.nan
    match = re.search(r"(\d+\.?\d*)", str(val))
    return float(match.group(1)) if match else np.nan

for col in ['mileage', 'engine', 'max_power']:
    df[col] = df[col].apply(clean_numeric)

df.dropna(subset=['mileage', 'engine', 'max_power', 'seats', 'selling_price'], inplace=True)
df['car_age'] = (datetime.datetime.now().year - df['year']).clip(lower=0)

# --- HYBRID FEATURE ENGINEERING ---
df['brand'] = df['name'].str.split(' ').str[0]
df['model'] = df['name'].str.split(' ').str[1]

# Split FIRST to prevent Data Leakage
X_raw = df.drop('selling_price', axis=1)
y_raw = df['selling_price']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# 1. Target Encoding for Brand
brand_target_map = y_train.groupby(X_train_raw['brand']).mean().to_dict()
global_mean = y_train.mean()

# 2. Frequency Encoding for Model
model_freq_map = X_train_raw['model'].value_counts(normalize=True).to_dict()

def apply_encodings(data):
    data = data.copy()
    data['brand_target_enc'] = data['brand'].map(brand_target_map).fillna(global_mean)
    data['model_freq_enc'] = data['model'].map(model_freq_map).fillna(0)
    return data

# Fix 1: Fixed the naming mistake/overwrite bug
X_train_pre = apply_encodings(X_train_raw)
X_test_pre = apply_encodings(X_test_raw)

# Drop raw columns
cols_to_drop = ['name', 'torque', 'year', 'brand', 'model']
X_train_pre.drop(columns=cols_to_drop, inplace=True)
X_test_pre.drop(columns=cols_to_drop, inplace=True)

# One-Hot Encoding
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
X_train = pd.get_dummies(X_train_pre, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test_pre, columns=categorical_cols, drop_first=True).reindex(columns=X_train.columns, fill_value=0)

# --- MODEL COMPARISON & ROBUST SELECTION ---
# Fix 4: Use KFold for reproducible Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
    # Fix 3: Added eval_metric to XGBoost to avoid warnings
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='rmse')
}

model_stats = {}
best_cv_r2 = -np.inf
best_model_name = ""

print("🚀 Evaluating models using 5-Fold Cross-Validation...")

for name, model in models.items():
    # Perform Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    mean_cv_r2 = cv_scores.mean()
    
    # Train on full training set for final test evaluation
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    test_r2 = r2_score(y_test, test_preds)
    mae = mean_absolute_error(y_test, test_preds)
    
    model_stats[name] = {
        "CV_Mean_R2": round(mean_cv_r2, 4),
        "CV_Scores": [round(s, 3) for s in cv_scores],
        "Test_R2": round(test_r2, 4),
        "MAE": round(mae, 2)
    }
    
    print(f"📊 {name:17} | CV R2: {mean_cv_r2:.4f} | Test R2: {test_r2:.4f}")
    
    # Fix 2: Best model selection based on CV Score for robustness
    if mean_cv_r2 > best_cv_r2:
        best_cv_r2 = mean_cv_r2
        best_model_name = name

# --- SAVING ARTIFACTS ---
joblib.dump(models[best_model_name], 'car_price_model.pkl')
joblib.dump(ui_options := {
    'brand': sorted(df['brand'].unique().tolist()),
    'model': sorted(df['model'].unique().tolist()),
    'fuel': sorted(df['fuel'].unique().tolist()),
    'seller_type': sorted(df['seller_type'].unique().tolist()),
    'transmission': sorted(df['transmission'].unique().tolist()),
    'owner': sorted(df['owner'].unique().tolist()),
    'seats': sorted([int(s) for s in df['seats'].unique()])
}, 'ui_options.pkl')

joblib.dump(brand_target_map, 'brand_target_map.pkl')
joblib.dump(model_freq_map, 'model_freq_map.pkl')
joblib.dump(global_mean, 'global_mean_price.pkl')
joblib.dump(model_stats, 'model_comparison.pkl')
joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')

if best_model_name in ["Random Forest", "XGBoost", "Decision Tree"]:
    feat_imp = pd.Series(models[best_model_name].feature_importances_, index=X_train.columns).to_dict()
    joblib.dump(feat_imp, 'feature_importance.pkl')

print(f"\n🏆 Champion Model: {best_model_name} (Selected via Cross-Validation)")
print(f"✅ Final Artifacts saved. Ready for Streamlit deployment.")