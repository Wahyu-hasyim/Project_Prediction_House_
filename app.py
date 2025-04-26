import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model & preprocessors
@st.cache_data
def load_artifacts(path='house_price_rf.pkl'):
    return joblib.load(path)

art = load_artifacts()
model       = art['model']
num_cols    = art['num_cols']
cat_cols    = art['cat_cols']
num_imputer = art['num_imputer']
scaler      = art['scaler']
cat_imputer = art['cat_imputer']
ord_enc     = art['ord_enc']

# Load original data for input defaults
@st.cache_data
def load_original(path='house_prices.csv'):
    return pd.read_csv(path)
orig_df = load_original()

st.title("House Price Predictor")


# Manual Input
st.header("Input Data")
input_dict = {}
cols = st.columns(2)
for i, col in enumerate(num_cols):
    with cols[i % 2]:
        min_val = float(orig_df[col].min())
        max_val = float(orig_df[col].max())
        median_val = float(orig_df[col].median())
        input_dict[col] = st.number_input(
            label=col,
            min_value=min_val,
            max_value=max_val,
            value=median_val
        )

for col in cat_cols:
    options = orig_df[col].dropna().unique().tolist()
    input_dict[col] = st.selectbox(
        label=col,
        options=options
    )

input_df = pd.DataFrame([input_dict])
if st.button("Predict Price"):
    X_num = scaler.transform(num_imputer.transform(input_df[num_cols]))
    X_cat_raw = cat_imputer.transform(input_df[cat_cols])
    X_cat = ord_enc.transform(X_cat_raw)
    X_enc = np.hstack([X_num, X_cat])
    pred = model.predict(X_enc)
    st.success(f"Predicted Price: {pred[0]:,.2f}")