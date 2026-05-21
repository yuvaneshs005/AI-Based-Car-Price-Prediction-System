import streamlit as st
import pandas as pd
import joblib
import datetime

# --- 1. LOAD ALL ARTIFACTS ---
# These must exist in the same folder as this script
try:
    model = joblib.load('car_price_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    ui_options = joblib.load('ui_options.pkl')
    brand_map = joblib.load('brand_target_map.pkl')
    model_map = joblib.load('model_freq_map.pkl')
    global_mean = joblib.load('global_mean_price.pkl')
    importance_data = joblib.load('feature_importance.pkl')
    model_comparison = joblib.load('model_comparison.pkl')
except FileNotFoundError:
    st.error("⚠️ Error: Missing model artifacts. Please run 'train_model.py' first!")
    st.stop()

# --- 2. UI CONFIGURATION ---
st.set_page_config(page_title="AI-Based Car Price Prediction System", page_icon="🏎️", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .prediction-box { padding: 20px; border-radius: 10px; background-color: #ffffff; border-left: 5px solid #ff4b4b; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("🏎️ AI-Based Car Price Prediction System")
st.sidebar.markdown("**AI-Based Car Price Prediction System**")
menu = st.sidebar.radio("Navigation", ["Price Predictor", "Model Analytics", "About Project"])

# Sidebar Quick Stats
st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
st.sidebar.success("✅ Model: Active")
st.sidebar.info(f"🏆 Champion: {max(model_comparison, key=lambda x: model_comparison[x]['CV_Mean_R2'])}")

# --- 4. PAGE: PRICE PREDICTOR ---
if menu == "Price Predictor":
    st.title("🚗 Smart Vehicle Valuation")
    st.write("Enter the vehicle specifications below to estimate the current market resale value.")

    # Input Form Layout
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🏷️ Basic Details")
            brand = st.selectbox("Car Brand", ui_options['brand'])
            car_model = st.selectbox("Model / Variant", ui_options['model'])
            year = st.number_input("Year of Manufacture", min_value=1990, max_value=datetime.datetime.now().year, value=2018)
            km_driven = st.number_input("Total KMs Driven", min_value=0, value=35000, step=1000)
            transmission = st.selectbox("Transmission", ui_options['transmission'])
            owner = st.selectbox("Owner History", ui_options['owner'])

        with col2:
            st.markdown("##### ⚙️ Technical Specs")
            fuel = st.selectbox("Fuel Type", ui_options['fuel'])
            engine = st.number_input("Engine Capacity (CC)", min_value=600, max_value=7000, value=1200)
            max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=1000.0, value=85.0)
            mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, value=18.0)
            seats = st.selectbox("Number of Seats", ui_options['seats'], index=ui_options['seats'].index(5))
            seller_type = st.selectbox("Seller Type", ui_options['seller_type'])

    st.markdown("---")
    
    if st.button("🚀 Calculate Resale Value"):
        # 1. Prepare Input DataFrame
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # 2. Apply Hybrid Encodings (Consistent with training logic)
        input_df['brand_target_enc'] = brand_map.get(brand, global_mean)
        input_df['model_freq_enc'] = model_map.get(car_model, 0)
        
        # 3. Numeric Mapping
        input_df['km_driven'] = km_driven
        input_df['mileage'] = mileage
        input_df['engine'] = engine
        input_df['max_power'] = max_power
        input_df['seats'] = seats
        input_df['car_age'] = max(0, datetime.datetime.now().year - year)
        
        # 4. One-Hot Encoding Mapping
        selections = {'fuel': fuel, 'seller_type': seller_type, 'transmission': transmission, 'owner': owner}
        for category, value in selections.items():
            dummy_col = f"{category}_{value}"
            if dummy_col in model_columns:
                input_df[dummy_col] = 1

        # 5. Prediction with Safety Floor
        raw_prediction = model.predict(input_df)[0]
        final_price = max(10000, raw_prediction) # Minimum 10k for any car
        
        # 6. Display Results
        st.markdown(f"""
            <div class="prediction-box">
                <h3 style='margin:0;'>Estimated Market Value</h3>
                <h1 style='color:#ff4b4b; margin:0;'>₹{final_price:,.2f}</h1>
                <p style='color:gray; font-size:0.9em;'>*Value based on current market trends and historical data.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.info(f"💡 **Data Insight:** {brand} vehicles in our database have a weighted average value of ₹{brand_map.get(brand, global_mean):,.2f}.")

# --- 5. PAGE: MODEL ANALYTICS ---
elif menu == "Model Analytics":
    st.title("📊 Technical Performance & Validation")
    
    # Section 1: Comparison Table
    st.subheader("1. Multi-Model Benchmarking (5-Fold CV)")
    st.write("We evaluated 4 algorithms. The champion was selected based on the highest Mean Cross-Validation R² score.")
    
    comp_df = pd.DataFrame(model_comparison).T
    # Highlight the best CV score
    st.dataframe(comp_df.style.highlight_max(axis=0, subset=['CV_Mean_R2'], color='#d4edda'))

    # Section 2: Feature Importance
    st.markdown("---")
    st.subheader("2. Global Feature Importance")
    st.write("Which features influence the price the most according to the Champion Model?")
    
    feat_df = pd.Series(importance_data).sort_values(ascending=False).head(8)
    st.bar_chart(feat_df)
    st.caption("Higher values indicate that the feature has more 'weight' in determining the car's final price.")

# --- 6. PAGE: ABOUT PROJECT ---
elif menu == "About Project":
    st.title("ℹ️ Project Documentation")
    st.markdown("""
    ### **AI-Based Car Price Prediction**
    This project was developed as a final-year technical milestone to solve the problem of high-cardinality categorical data in car price prediction.

    #### **Key Technical Highlights:**
    * **Hybrid Encoding:** Used a combination of Target Encoding (for Brands) and Frequency Encoding (for Models) to handle 2000+ unique car names without overfitting.
    * **Anti-Leakage Pipeline:** Feature engineering was performed strictly after the train-test split to ensure realistic performance.
    * **Cross-Validation:** 5-Fold K-Fold Cross-Validation was used to ensure model stability across different data subsets.
    * **Explainable AI:** Integrated feature importance visualization to provide transparency in pricing logic.
    """)
    
    st.markdown("---")
    st.write("Developed by **Yuvanesh S**")