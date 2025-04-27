import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import cloudpickle

# ————————————
# Caching Functions
# ————————————
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'Price' not in df.columns:
        if 'Price (in rupees)' in df.columns:
            df['Price'] = pd.to_numeric(df['Price (in rupees)'], errors='coerce')
        else:
            raise KeyError("Kolom 'Price' atau 'Price (in rupees)' tidak ditemukan.")
    return df

@st.cache_resource(show_spinner="Memuat model/artifacts...")
def load_artifacts(path: str):
    try:
        return joblib.load(path)
    except ModuleNotFoundError:
        with open(path, 'rb') as f:
            return cloudpickle.load(f)

# ————————————
# Render Halaman Visualisasi
# ————————————
def render_visualisasi():
    st.title("Visualisasi Data dan Model Interaktif")

    # Load data
    try:
        df = load_data('house_prices.csv')
    except FileNotFoundError:
        st.error("Dataset 'house_prices.csv' tidak ditemukan.")
        return
    except KeyError as e:
        st.error(str(e))
        return

    # Load model/artifacts (optional)
    try:
        artifacts = load_artifacts('house_price_rf.pkl')
    except FileNotFoundError:
        st.warning("Model/artifact 'house_price_rf.pkl' tidak ditemukan. Hanya visualisasi yang dijalankan.")
        artifacts = None

    # Sidebar: Filter & Input
    st.sidebar.header("Filter & Pengaturan Grafis")
    min_price, max_price = int(df['Price'].min()), int(df['Price'].max())
    price_range = st.sidebar.slider(
        "Pilih rentang Harga Rumah:",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=1_000_000
    )
    df_filt = df.query("@price_range[0] <= Price <= @price_range[1]")

    num_cols = df.select_dtypes(include='number').columns.tolist()
    selected_feature = st.sidebar.selectbox(
        "Fitur untuk distribusi:",
        options=num_cols,
        index=num_cols.index('Price') if 'Price' in num_cols else 0
    )
    nbins = st.sidebar.slider("Jumlah bins histogram:", 10, 100, 50, 5)
    selected_corr = st.sidebar.multiselect("Fitur untuk heatmap korelasi:", options=num_cols, default=num_cols)

    # 1. Distribusi
    st.subheader(f"1. Distribusi `{selected_feature}`")
    fig_dist = px.histogram(
        df_filt,
        x=selected_feature,
        nbins=nbins,
        title=f"Distribusi {selected_feature} (Rentang {price_range[0]:,} – {price_range[1]:,})",
        marginal='box'
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # 2. Heatmap Korelasi
    if len(selected_corr) >= 2:
        st.subheader("2. Heatmap Korelasi Fitur Numerik")
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

# ————————————
# Entry Point
# ————————————
if __name__ == '__main__':
    render_visualisasi()
