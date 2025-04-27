import streamlit as st

# 1) set_page_config harus di baris pertama Streamlit command
st.set_page_config(
    page_title="Aplikasi DS",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2) Baru kemudian definisikan pages dan navigasi
pages = [
    st.Page("home.py",          title="Home"),
    st.Page("app.py",           title="Aplikasi"),
    st.Page("visualisasi.py",   title="Dashboard"),
    st.Page("tentang_saya.py",  title="Tentang Saya"),
]

# 3) Buat navigasi dan jalankan halaman terpilih
pg = st.navigation(pages)
pg.run()
