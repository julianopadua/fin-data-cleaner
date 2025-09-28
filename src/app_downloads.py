# src/app_downloads.py
import os
import io
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Iterable, Any

import pandas as pd
import streamlit as st

from scrapers import load_config, list_datasets, download_dataset
from analysis import (
    describe_table_basic,
    profile_dataframe,
    profiles_to_dataframe,
)

# ----------------------------
# Page setup & lightweight CSS
# ----------------------------

def setup_page() -> None:
    st.set_page_config(page_title="FinData Cleaner - Downloads", layout="wide")
    st.title("FinData Cleaner - Downloads")
    st.markdown(
        """
        <style>
          .fin-title { font-size: 1.15rem; font-weight: 700; margin: 0 0 .25rem 0; }
          .fin-help { font-size: 0.85rem; color: #6c757d; margin-top: -6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# Generic file & data helpers
# ----------------------------
_DATA_SHEET_RE = re.compile(r"^data\s*\d*$", re.IGNORECASE)

def _excel_engine_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    # .xls precisa xlrd<=1.2.0; .xlsx/.xlsm usa openpyxl
    return "xlrd" if ext == ".xls" else "openpyxl"

def _list_data_sheets(path: str) -> List[str]:
    engine = _excel_engine_for(path)
    xls = pd.ExcelFile(path, engine=engine)
    sheets = [s for s in xls.sheet_names if _DATA_SHEET_RE.match(s.strip())]
    return sheets or xls.sheet_names  # fallback: mostra todas se não achar "Data ..."


def clear_dir_files(dir_path: str) -> None:
    """Delete files in a directory (non-recursive). Keeps subfolders if any."""
    if not os.path.isdir(dir_path):
        return
    for name in os.listdir(dir_path):
        fp = os.path.join(dir_path, name)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except Exception:
                pass  # stay silent in UI

def _try_read_excel(path: str, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Best-effort reader for legacy .xls (EIA usa Excel antigo).
    Tenta engine='xlrd' se disponível. Não quebra a UI se falhar.
    """
    try:
        # pandas usará xlrd para .xls se instalado (<2.0). Caso falhe, capturamos.
        if nrows is None:
            return pd.read_excel(path, dtype=str)
        return pd.read_excel(path, dtype=str, nrows=nrows)
    except Exception:
        return None

def preview_eia_excel_data_tabs(path: str, filename: str, head_n: int = 100, header_row_idx: int = 2) -> None:
    """
    Lê cada aba "Data ..." usando a linha 3 (índice 2) como header.
    Faz limpeza leve e exibe preview + download em CSV.
    """
    engine = _excel_engine_for(path)
    data_sheets = _list_data_sheets(path)

    st.markdown("<div class='fin-title'>Pré-visualização das abas \"Data\"</div>", unsafe_allow_html=True)
    tabs = st.tabs(data_sheets)

    def _fix_cols(cols: Iterable[Any]) -> List[str]:
        fixed: List[str] = []
        seen: set[str] = set()
        for i, c in enumerate(cols):
            name = (str(c).strip() if c is not None else "").strip()
            if not name or name.lower().startswith("unnamed"):
                name = f"col_{i}"
            # de-dup para colunas repetidas
            base = name
            k = 1
            while name in seen:
                k += 1
                name = f"{base}_{k}"
            seen.add(name)
            fixed.append(name)
        return fixed

    for tab, sheet in zip(tabs, data_sheets):
        with tab:
            try:
                # header_row_idx=2 => 3ª linha é o cabeçalho
                df = pd.read_excel(path, sheet_name=sheet, engine=engine, header=header_row_idx)
            except Exception as e:
                st.error(f"Falha ao ler a aba {sheet} com header na linha 3: {e}")
                # fallback conservador: tenta com header padrão
                try:
                    df = pd.read_excel(path, sheet_name=sheet, engine=engine)
                    st.info("Não foi possível usar a linha 3 como cabeçalho nesta aba. Foi aplicado um fallback de leitura padrão.")
                except Exception as e2:
                    st.error(f"Falha no fallback ao ler a aba {sheet}: {e2}")
                    continue

            # limpeza leve de linhas/colunas totalmente vazias
            df = df.dropna(how="all").dropna(axis=1, how="all")

            # normaliza nomes das colunas
            df.columns = _fix_cols(df.columns)

            st.caption(f"{filename} • aba: {sheet} • {df.shape[0]} linhas × {df.shape[1]} colunas")
            st.dataframe(df.head(head_n), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Baixar CSV desta aba",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{os.path.splitext(filename)[0]}_{sheet.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"dl_csv_{sheet}"
                )
            with c2:
                if st.checkbox("Mostrar estatísticas desta aba", key=f"stats_{sheet}"):
                    st.write(df.describe(include="all", percentiles=[.25, .5, .75]).T)


def read_preview_table(path: str, nrows: int = 200) -> Optional[pd.DataFrame]:
    """Read a small preview (tries common separators/encodings and .xls)."""
    ext = path.split(".")[-1].lower()
    if ext in {"txt", "csv"}:
        for enc in ("utf-8", "latin-1"):
            for sep in (";", ","):
                try:
                    return pd.read_csv(path, sep=sep, nrows=nrows, dtype=str, encoding=enc)
                except Exception:
                    continue
        return None
    if ext in {"xls"}:
        return _try_read_excel(path, nrows=nrows)
    return None  # extend here for other formats

def read_full_table(path: str) -> Optional[pd.DataFrame]:
    """Read full table for conversion or stats (safe parsing)."""
    ext = path.split(".")[-1].lower()
    if ext in {"txt", "csv"}:
        for enc in ("utf-8", "latin-1"):
            for sep in (";", ","):
                try:
                    return pd.read_csv(path, sep=sep, dtype=str, encoding=enc)
                except Exception:
                    continue
        return None
    if ext in {"xls"}:
        return _try_read_excel(path, nrows=None)
    return None

def build_download_payload(sel_path: str, filename: str, choice: str) -> Tuple[Optional[bytes], str, str]:
    """
    Build bytes + mime + download_filename according to user choice.
    choice can be 'CSV', 'XLSX' or '<EXT> (arquivo original)'.
    """
    import pandas as pd  # local import to avoid circulars on some setups

    orig_ext = (filename.split(".")[-1].lower() if "." in filename else "txt")
    opt_original = f"{orig_ext.upper()} (arquivo original)"

    if choice == opt_original:
        try:
            with open(sel_path, "rb") as f:
                payload = f.read()
            return payload, "application/octet-stream", filename
        except Exception:
            return None, "application/octet-stream", filename

    df_full = read_full_table(sel_path)
    if df_full is None:
        # Fallback to original if we couldn't parse
        try:
            with open(sel_path, "rb") as f:
                payload = f.read()
            return payload, "application/octet-stream", filename
        except Exception:
            return None, "application/octet-stream", filename

    if choice == "CSV":
        payload = df_full.to_csv(index=False).encode("utf-8")
        dl_name = os.path.splitext(filename)[0] + ".csv"
        return payload, "text/csv", dl_name

    if choice == "XLSX":
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df_full.to_excel(writer, index=False, sheet_name="dados")
        payload = bio.getvalue()
        dl_name = os.path.splitext(filename)[0] + ".xlsx"
        return payload, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", dl_name

    return None, "application/octet-stream", filename


# ----------------------------
# Shared UI blocks
# ----------------------------

def controls_row(filename: str, sel_path: str) -> Tuple[str, Optional[bytes], str, str, bool]:
    """
    Render the controls above the table:
    - left: file type selector (big title)
    - right: 'Ver estatísticas' (top) and 'Baixar' (bottom)
    Returns (choice, payload_bytes, mime, dl_name, view_stats_clicked).
    """
    orig_ext = (filename.split(".")[-1].lower() if "." in filename else "txt")
    opt_original = f"{orig_ext.upper()} (arquivo original)"
    options = [opt_original, "CSV", "XLSX"]

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.markdown("<div class='fin-title'>Tipo de arquivo para download</div>", unsafe_allow_html=True)
        choice = st.selectbox("", options, index=0, label_visibility="collapsed")

    with col_right:
        stats_clicked = st.button("Ver estatísticas", use_container_width=True, key="btn_stats")
        payload, mime, dl_name = build_download_payload(sel_path, filename, choice)
        st.download_button(
            "Baixar",
            data=payload if payload is not None else b"",
            file_name=dl_name,
            mime=mime,
            use_container_width=True,
            key="btn_download_payload",
            disabled=(payload is None),
        )

    return choice, payload, mime, dl_name, stats_clicked

def render_preview_and_optional_stats(sel_path: str, filename: str, show_stats: bool) -> None:
    """Show preview table and, if requested, descriptive stats using analysis.py."""
    df_head = read_preview_table(sel_path, nrows=200)
    st.subheader(f"Pré-visualização: {filename}")

    if df_head is not None:
        df_head.columns = [c.strip() if isinstance(c, str) else c for c in df_head.columns]
        st.dataframe(df_head.head(5), use_container_width=True)
        st.caption(f"Linhas lidas (amostra): {len(df_head):,} • Colunas: {df_head.shape[1]}")

        if show_stats:
            df_full = read_full_table(sel_path)
            if df_full is None:
                st.warning("Não consegui interpretar a base completa para estatísticas.")
                return

            st.divider()
            st.subheader("Informações da tabela")
            info = describe_table_basic(df_full)
            c1, c2, c3 = st.columns(3)
            c1.metric("Linhas (total)", f"{info['rows']:,}")
            c2.metric("Colunas (total)", f"{info['cols']:,}")
            c3.metric("Memória (MB)", f"{info['memory_mb']}")

            st.subheader("Estatísticas por coluna")
            prof = profile_dataframe(df_full, max_categorical_unique=20)
            st.dataframe(profiles_to_dataframe(prof), use_container_width=True)
    else:
        st.info("Pré-visualização automática não disponível para este formato. O arquivo já está salvo; você pode baixá-lo para abrir localmente.")


# ----------------------------
# Provider: CONAB (file downloads)
# ----------------------------

def conab_flow(cfg: dict) -> None:
    """Original CONAB flow: list page datasets, download chosen file, preview, stats, download conversion."""
    # left/right selection
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("<div class='fin-title'>Base de dados</div>", unsafe_allow_html=True)
        st.write("CONAB - Downloads de Arquivos")

    with st.spinner("Carregando lista de datasets..."):
        items = list_datasets("conab", cfg)
    if not items:
        st.warning("Nenhum dataset encontrado para esta base.")
        return

    labels = {
        d["id"]: f"{d['category']} • {d['name']} ({d['filename']})" for d in items
    }
    ids_sorted = sorted(labels.keys(), key=lambda i: labels[i])

    with col_b:
        st.markdown("<div class='fin-title'>Qual dataset você quer acessar</div>", unsafe_allow_html=True)
        dataset_label = st.selectbox("", [labels[i] for i in ids_sorted], label_visibility="collapsed")
        label_to_id = {labels[i]: i for i in ids_sorted}
        chosen_id = label_to_id[dataset_label]
        clicked = st.button("Selecionar", type="primary", use_container_width=True, key="btn_select_conab")

    if clicked:
        with st.spinner("Limpando diretório e baixando a base selecionada..."):
            data_dir = cfg["paths"]["data_raw"]
            clear_dir_files(data_dir)
            path = download_dataset("conab", chosen_id, cfg, overwrite=True)
            st.session_state["_selected_path"] = path
            st.session_state["_selected_filename"] = os.path.basename(path)

    st.divider()

    if "_selected_path" in st.session_state:
        sel_path = st.session_state["_selected_path"]
        filename = st.session_state.get("_selected_filename", os.path.basename(sel_path))
        _, _, _, _, stats_clicked = controls_row(filename, sel_path)
        render_preview_and_optional_stats(sel_path, filename, show_stats=stats_clicked)


# ----------------------------
# Provider: EIA - Petroleum Supply Monthly
# ----------------------------

def eia_psm_flow(cfg: dict) -> None:
    """
    Passo 1: escolher a seção (categoria).
    Passo 2: escolher a série pelo NOME (sem exibir id/filename).
    Ação única: Baixar XLS; depois disso mostramos a pré-visualização local.
    """
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("<div class='fin-title'>Base de dados</div>", unsafe_allow_html=True)
        st.write("EIA – Petroleum Supply Monthly")

    with st.spinner("Carregando lista de séries do PSM..."):
        items = list_datasets("eia", cfg)  # alias p/ eia_psm
    if not items:
        st.warning("Nenhuma série encontrada para EIA PSM.")
        return

    # Agrupar por categoria (thead)
    def _cat_of(d: Dict) -> str:
        cat = (d.get("category") or "").strip()
        return cat if cat else "Outros"

    by_cat: Dict[str, List[Dict]] = {}
    for d in items:
        by_cat.setdefault(_cat_of(d), []).append(d)

    cats_sorted = sorted(by_cat.keys())

    # PASSO 1: categoria
    with col_b:
        st.markdown("<div class='fin-title'>Selecione a seção</div>", unsafe_allow_html=True)
        chosen_cat = st.selectbox(
            "",
            options=cats_sorted,
            index=(cats_sorted.index("Country of Origin") if "Country of Origin" in cats_sorted else 0),
            label_visibility="collapsed",
            help="Ex.: Country of Origin, Crude Oil, Refinery Operations, PAD District, ..."
        )

    # PASSO 2: série (somente nome visível)
    series_items = by_cat.get(chosen_cat, [])
    series_items.sort(key=lambda d: (d.get("name") or "").lower())
    series_names = [d.get("name") or d.get("id") for d in series_items]

    st.markdown("<div class='fin-title'>Selecione a série</div>", unsafe_allow_html=True)
    choice_name = st.selectbox("", series_names, label_visibility="collapsed")

    # localizar item escolhido (para obter o id)
    name_to_item = { (d.get("name") or d.get("id")): d for d in series_items }
    selected_item = name_to_item[choice_name]
    chosen_id = selected_item["id"]

    # Ação única: baixar
    if st.button("Baixar XLS da série selecionada", type="primary", use_container_width=True, key="btn_select_eia_psm"):
        with st.spinner("Limpando diretório e baixando o XLS da série..."):
            data_dir = cfg["paths"]["data_raw"]
            clear_dir_files(data_dir)
            path = download_dataset("eia", chosen_id, cfg, overwrite=True)
            st.session_state["_selected_path"] = path
            st.session_state["_selected_filename"] = os.path.basename(path)

    st.divider()

    # Pós-download: preview local + estatísticas e conversões
    if "_selected_path" in st.session_state:
        sel_path = st.session_state["_selected_path"]
        filename = st.session_state.get("_selected_filename", os.path.basename(sel_path))

        # (opcional) mantém seus controles existentes:
        # _, _, _, _, stats_clicked = controls_row(filename, sel_path)

        # preview por abas "Data ..."
        preview_eia_excel_data_tabs(sel_path, filename)


# ----------------------------
# Router: choose provider, run flow
# ----------------------------

def select_provider() -> str:
    st.markdown("<div class='fin-title'>Base de dados</div>", unsafe_allow_html=True)
    providers = {
        "CONAB - Downloads de Arquivos": "conab",
        "EIA - Petroleum Supply Monthly": "eia_psm",
    }
    label = st.selectbox("", list(providers.keys()), label_visibility="collapsed")
    return providers[label]

def main() -> None:
    setup_page()
    cfg = load_config("config.yaml")

    provider = select_provider()
    st.divider()

    if provider == "conab":
        conab_flow(cfg)
    elif provider == "eia_psm":
        eia_psm_flow(cfg)

if __name__ == "__main__":
    main()
