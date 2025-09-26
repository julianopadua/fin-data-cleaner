# src/app_downloads.py
import io
import csv
import pandas as pd
import streamlit as st

from scrapers import load_config, list_datasets, download_dataset

st.set_page_config(page_title="FinData Cleaner – Downloads", layout="wide")
st.title("FinData Cleaner – Downloads")

cfg = load_config("config.yaml")

# 1) Escolher fornecedor (base)
providers = {
    "CONAB – Downloads de Arquivos": "conab",
    # amanhã: "IBGE – SIDRA": "sidra", "B3 – Dados Históricos": "b3", etc.
}
provider_label = st.selectbox("Base de dados:", list(providers.keys()))
provider = providers[provider_label]

# 2) Listar datasets do fornecedor
with st.spinner("Carregando lista de datasets..."):
    items = list_datasets(provider, cfg)

if not items:
    st.warning("Nenhum dataset encontrado para esta base.")
    st.stop()

# montar rótulo legível
labels = {
    d["id"]: f"{d['category']} • {d['name']} ({d['filename']})" for d in items
}
ids_sorted = sorted(labels.keys(), key=lambda i: labels[i])
dataset_id = st.selectbox("Qual dataset você quer acessar?", [labels[i] for i in ids_sorted])
# mapear label -> id
label_to_id = {labels[i]: i for i in ids_sorted}
chosen_id = label_to_id[dataset_id]

col_a, col_b = st.columns([1,1], gap="large")
with col_a:
    if st.button("Selecionar", type="primary", use_container_width=True):
        with st.spinner("Baixando e preparando pré-visualização..."):
            path = download_dataset(provider, chosen_id, cfg, overwrite=False)
            st.session_state["_selected_path"] = path
            st.success(f"Arquivo disponível em: {path}")

with col_b:
    if st.button("Baixar novamente (overwrite)", use_container_width=True):
        with st.spinner("Rebaixando arquivo..."):
            path = download_dataset(provider, chosen_id, cfg, overwrite=True)
            st.session_state["_selected_path"] = path
            st.success(f"Arquivo substituído em: {path}")

# 3) Pré-visualização do arquivo selecionado
def _try_read_table(path: str, nrows: int | None = None) -> pd.DataFrame | None:
    ext = path.split(".")[-1].lower()
    if ext in {"txt", "csv"}:
        # tentativa 1: separador ';'
        try:
            return pd.read_csv(path, sep=";", nrows=nrows, dtype=str, encoding="utf-8")
        except Exception:
            pass
        # tentativa 2: separador ','
        try:
            return pd.read_csv(path, sep=",", nrows=nrows, dtype=str, encoding="utf-8")
        except Exception:
            pass
        # tentativa 3: encodings latinos
        try:
            return pd.read_csv(path, sep=";", nrows=nrows, dtype=str, encoding="latin-1")
        except Exception:
            try:
                return pd.read_csv(path, sep=",", nrows=nrows, dtype=str, encoding="latin-1")
            except Exception:
                return None
    # outros formatos (xls/xlsx/parquet) – você pode expandir aqui futuramente
    return None

st.divider()
if "_selected_path" in st.session_state:
    sel_path = st.session_state["_selected_path"]
    st.subheader("Pré-visualização")
    df_head = _try_read_table(sel_path, nrows=200)  # lê amostra maior e mostra head(5)
    if df_head is not None:
        # limpar cabeçalhos (trim) e mostrar
        df_head.columns = [c.strip() if isinstance(c, str) else c for c in df_head.columns]
        st.write(f"Arquivo: `{sel_path}`")
        st.dataframe(df_head.head(5), use_container_width=True)
        st.caption(f"Linhas lidas (amostra): {len(df_head):,} • Colunas: {df_head.shape[1]}")
        with st.expander("Ver nomes das colunas"):
            st.code("\n".join(map(str, df_head.columns.tolist())))
    else:
        st.info("Pré-visualização automática não disponível para este formato. O arquivo já está salvo; você pode baixá-lo para abrir localmente.")
