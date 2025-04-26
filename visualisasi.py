import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- Page Title ---
st.title("Visualisasi Data dan Model Interaktif")

# --- Load Data ---
try:
    df = pd.read_csv('house_prices.csv')
except FileNotFoundError:
    st.error("Dataset 'house_prices.csv' tidak ditemukan di direktori aplikasi.")
    st.stop()
if 'Price' not in df.columns:
    if 'Price (in rupees)' in df.columns:
        df['Price'] = pd.to_numeric(df['Price (in rupees)'], errors='coerce')
    else:
        st.error("Kolom 'Price' atau 'Price (in rupees)' tidak ditemukan.")
        st.stop()

# Load Model Artifacts (jika diperlukan nanti) ---
try:
    artifacts = joblib.load('house_price_rf.pkl')
except FileNotFoundError:
    st.warning("File model 'house_price_rf.pkl' tidak ditemukan. Hanya visualisasi yang dijalankan.")
    artifacts = None

# Widget Input untuk Interaksi ---

st.subheader("Filter Data dan Pengaturan Grafik")

# 0.1 Slider untuk rentang harga
min_price, max_price = int(df['Price'].min()), int(df['Price'].max())
price_range = st.slider(
    "Pilih rentang Harga Rumah:",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=1000000
)
df_filt = df[(df['Price'] >= price_range[0]) & (df['Price'] <= price_range[1])]

# 0.2 Pilih fitur numerik untuk histogram
num_cols = df.select_dtypes(include='number').columns.tolist()
selected_feature = st.selectbox(
    "Pilih fitur untuk distribusi:",
    options=num_cols,
    index=num_cols.index('Price')
)

# 0.3 Atur jumlah bins
nbins = st.slider(
    "Jumlah bins histogram:",
    min_value=10,
    max_value=100,
    value=50,
    step=5
)

# 0.4 Pilih subset fitur untuk korelasi
st.markdown("**Pilih fitur numerik untuk heatmap korelasi:**")
selected_corr = st.multiselect(
    label="Fitur numerik:",
    options=num_cols,
    default=num_cols
)

# --- 1. Distribusi Harga (atau fitur terpilih) ---

st.subheader(f"1. Distribusi `{selected_feature}`")
fig_price = px.histogram(
    df_filt,
    x=selected_feature,
    nbins=nbins,
    title=f'Distribusi {selected_feature} (Rentang {price_range[0]:,} – {price_range[1]:,})',
    marginal='box'
)
st.plotly_chart(fig_price, use_container_width=True)

# --- 2. Heatmap Korelasi Fitur Numerik ---

if len(selected_corr) >= 2:
    st.subheader("2. Korelasi Fitur Numerik")
    corr_matrix = df_filt[selected_corr].corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect='auto',
        title='Heatmap Korelasi'
    )
    fig_corr.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Pilih minimal 2 fitur untuk menampilkan heatmap korelasi.")