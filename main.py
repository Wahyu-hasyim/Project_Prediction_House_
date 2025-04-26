import streamlit as st

pages = [
    st.Page("home.py",              title="Home"),
    st.Page("app.py",                title="Aplikasi"),
    st.Page("visualisasi.py",  title="Dashboard"),
    st.Page("tentang_saya.py",          title="Tentang saya"),
]

st.set_page_config(page_title="Aplikasi DS", page_icon="🚀")

#Buat navigasi dan jalankan halaman terpilih
pg = st.navigation(pages)
pg.run()