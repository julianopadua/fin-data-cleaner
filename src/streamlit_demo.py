# src/app_downloads.py
import os 
import io
import csv
import pandas as pd
import streamlit as st

from scrapers import load_config, list_datasets, download_dataset

st.set_page_config(page_title="FinData Cleaner – Downloads", layout="wide")
st.title("FinData Cleaner – Downloads")

# estilos para títulos dos selects
st.markdown("""
<style>
.fin-title { font-size: 1.15rem; font-weight: 700; margin: 0 0 .25rem 0; }
</style>
""", unsafe_allow_html=True)

cfg = load_config("config.yaml")

# 1) Escolher fornecedor (base) e dataset em duas colunas
providers = {
    "CONAB – Downloads de Arquivos": "conab",
    # amanhã: "IBGE – SIDRA": "sidra", "B3 – Dados Históricos": "b3", etc.
}

col_sel_a, col_sel_b = st.columns([1, 1], gap="large")

with col_sel_a:
    st.markdown("<div class='fin-title'>Base de dados</div>", unsafe_allow_html=True)
    provider_label = st.selectbox("", list(providers.keys()), label_visibility="collapsed")
    provider = providers[provider_label]

# carregar itens depois de escolher o provider
with st.spinner("Carregando lista de datasets..."):
    items = list_datasets(provider, cfg)

if not items:
    st.warning("Nenhum dataset encontrado para esta base.")
    st.stop()

# montar rótulos legíveis
labels = {d["id"]: f"{d['category']} • {d['name']} ({d['filename']})" for d in items}
ids_sorted = sorted(labels.keys(), key=lambda i: labels[i])

with col_sel_b:
    st.markdown("<div class='fin-title'>Qual dataset você quer acessar</div>", unsafe_allow_html=True)
    dataset_label = st.selectbox("", [labels[i] for i in ids_sorted], label_visibility="collapsed")
    label_to_id = {labels[i]: i for i in ids_sorted}
    chosen_id = label_to_id[dataset_label]
    select_clicked = st.button("Selecionar", type="primary", use_container_width=True, key="btn_select")

if select_clicked:
    with st.spinner("Baixando e preparando pré-visualização..."):
        path = download_dataset(provider, chosen_id, cfg, overwrite=False)
        st.session_state["_selected_path"] = path
        st.session_state["_selected_filename"] = os.path.basename(path)


def _read_full_table(path: str) -> pd.DataFrame | None:
    ext = path.split(".")[-1].lower()
    if ext in {"txt", "csv"}:
        for enc in ("utf-8", "latin-1"):
            for sep in (";", ","):
                try:
                    return pd.read_csv(path, sep=sep, dtype=str, encoding=enc)
                except Exception:
                    continue
    return None


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
    filename = st.session_state.get("_selected_filename", os.path.basename(sel_path))
    st.subheader("Pré-visualização")
    df_head = _try_read_table(sel_path, nrows=200)  # lê amostra maior e mostra head(5)
    if df_head is not None:
        # limpar cabeçalhos (trim) e mostrar
        df_head.columns = [c.strip() if isinstance(c, str) else c for c in df_head.columns]
        st.write(f"Arquivo: `{filename}`")
        st.dataframe(df_head.head(5), use_container_width=True)
        st.caption(f"Linhas lidas (amostra): {len(df_head):,} • Colunas: {df_head.shape[1]}")
        with st.expander("Ver nomes das colunas"):
            st.code("\n".join(map(str, df_head.columns.tolist())))
    else:
        st.info("Pré-visualização automática não disponível para este formato. O arquivo já está salvo; você pode baixá-lo para abrir localmente.")
    # --- opções de download ---
    orig_ext = (filename.split(".")[-1].lower() if "." in filename else "txt")
    opt_original = f"{orig_ext.upper()} (arquivo original)"
    download_opts = [opt_original, "CSV", "XLSX"]

    dl_col_a, dl_col_b = st.columns([2, 1], gap="large")
    with dl_col_a:
        dl_choice = st.selectbox("Tipo de arquivo para download", download_opts, index=0)

    # preparar payload
    payload_bytes = None
    mime = "application/octet-stream"
    dl_name = filename

    if dl_choice == opt_original:
        try:
            with open(sel_path, "rb") as f:
                payload_bytes = f.read()
            mime = "application/octet-stream"
            dl_name = filename
        except Exception as e:
            st.error(f"Falha ao abrir arquivo original: {e}")
    else:
        df_full = _read_full_table(sel_path)
        if df_full is None:
            st.warning("Não consegui interpretar a base para conversão; baixando arquivo original.")
            with open(sel_path, "rb") as f:
                payload_bytes = f.read()
            mime = "application/octet-stream"
            dl_name = filename
        else:
            if dl_choice == "CSV":
                payload_bytes = df_full.to_csv(index=False).encode("utf-8")
                mime = "text/csv"
                dl_name = os.path.splitext(filename)[0] + ".csv"
            elif dl_choice == "XLSX":
                bio = io.BytesIO()
                with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                    df_full.to_excel(writer, index=False, sheet_name="dados")
                payload_bytes = bio.getvalue()
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                dl_name = os.path.splitext(filename)[0] + ".xlsx"

    with dl_col_b:
        if payload_bytes is not None:
            st.download_button(
                "Baixar",
                data=payload_bytes,
                file_name=dl_name,
                mime=mime,
                use_container_width=True,
                key="btn_download_payload",
            )
