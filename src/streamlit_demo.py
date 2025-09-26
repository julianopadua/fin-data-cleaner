# streamlit_demo.py
import streamlit as st
from scrapers import load_config, list_datasets, download_dataset

st.title("FinData Cleaner – Downloads")
cfg = load_config("config.yaml")

items = list_datasets("conab", cfg)
options = {f"{x['category']} • {x['name']} ({x['filename']})": x['id'] for x in items}
choice = st.selectbox("Selecione um dataset (CONAB):", list(options.keys()))

if st.button("Baixar"):
    path = download_dataset("conab", options[choice], cfg, overwrite=False)
    st.success(f"Arquivo salvo em: {path}")
    st.download_button("Download", data=open(path, "rb").read(), file_name=path.split("/")[-1])
