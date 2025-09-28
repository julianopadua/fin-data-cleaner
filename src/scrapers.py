"""
Scrapers module (extensible).
Current providers:
- "conab": https://portaldeinformacoes.conab.gov.br/download-arquivos.html

Quick usage (non-Streamlit):
    from scrapers import load_config, list_datasets, download_dataset

    cfg = load_config("config.yaml")
    items = list_datasets("conab", cfg)
    # items is a list of dicts: {id, provider, category, name, url, filename, ext}

    # pick one by id (e.g., items[0]["id"])
    saved_path = download_dataset("conab", items[0]["id"], cfg)
"""

from __future__ import annotations

import os
import re
import json
import logging
import unicodedata
from dataclasses import dataclass, asdict
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import yaml
from tqdm import tqdm
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


# ---------------------------
# config + paths + logging
# ---------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # sensible defaults
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("data_raw", "./data/raw")
    cfg["paths"].setdefault("data_processed", "./data/processed")
    cfg["paths"].setdefault("images", "./images")
    cfg["paths"].setdefault("report", "./report")
    cfg["paths"].setdefault("addons", "./addons")
    cfg["paths"].setdefault("logs", "./logs")

    cfg.setdefault("scraper", {})
    cfg["scraper"].setdefault("timeout_s", 30)
    cfg["scraper"].setdefault("retries", 3)
    cfg["scraper"].setdefault("backoff_factor", 1.0)
    cfg["scraper"].setdefault(
        "user_agent",
        "FinDataCleaner/0.1 (+https://example.com; contact: admin@example.com)",
    )
    # make sure folders exist
    for p in cfg["paths"].values():
        os.makedirs(p, exist_ok=True)
    return cfg


def _setup_logger(cfg: dict) -> logging.Logger:
    log_dir = cfg["paths"]["logs"]
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "scrapers.log")

    logger = logging.getLogger("scrapers")
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers during hot-reload
    if not logger.handlers:
        fh = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _requests_session(cfg: dict) -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=int(cfg["scraper"]["retries"]),
        connect=int(cfg["scraper"]["retries"]),
        read=int(cfg["scraper"]["retries"]),
        backoff_factor=float(cfg["scraper"]["backoff_factor"]),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.headers.update({"User-Agent": cfg["scraper"]["user_agent"], "Accept-Charset": "utf-8, iso-8859-1;q=0.8",
    "Accept-Language": "pt-BR, pt;q=0.9, en;q=0.8"})
    return sess


# ---------------------------
# models & helpers
# ---------------------------

@dataclass(frozen=True)
class DatasetLink:
    id: str
    provider: str
    category: str
    name: str
    url: str
    filename: str
    ext: str

    def to_dict(self) -> dict:
        return asdict(self)


def _slugify(s: str) -> str:
    # normalize accents -> ascii, keep alnum, dash, underscore
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9\-_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-").lower()
    return s

def _clean_unicode(s: str) -> str:
    # troca NBSP por espaço, normaliza composição (NFC) e tira espaços laterais
    return unicodedata.normalize("NFC", (s or "").replace("\xa0", " ")).strip()


def _filename_from_url(url: str) -> str:
    return os.path.basename(urlparse(url).path)


def _ext(name: str) -> str:
    m = re.search(r"\.([A-Za-z0-9]+)$", name)
    return m.group(1).lower() if m else ""


def _save_stream(session: requests.Session, url: str, dest_path: str, timeout: int, logger: logging.Logger) -> str:
    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        with open(dest_path, "wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {os.path.basename(dest_path)}",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    if total:
                        pbar.update(len(chunk))
    logger.info(f"Saved file: {dest_path}")
    return dest_path

# ---------------------------
# PROVIDER: EIA - Petroleum Supply Monthly (PSM)
# ---------------------------

_EIA_BASE = "https://www.eia.gov"
_EIA_PSM_INDEX = f"{_EIA_BASE}/petroleum/supply/monthly/"

def _eia_psm_parse_index(html: str) -> List[DatasetLink]:
    """
    Varre a tabela principal da página do PSM e coleta links para as páginas .htm
    das séries (cada uma contém o link 'Download Series History' para .xls).
    category = texto do <thead> mais próximo acima do link.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="basic-table")
    if not table:
        return []

    items: List[DatasetLink] = []
    for a in table.select("a[href]"):
        href = a.get("href", "").strip()
        if not (href.startswith("/dnav/pet/") and href.endswith(".htm")):
            continue

        series_page_url = urljoin(_EIA_BASE, href)
        series_id = os.path.splitext(os.path.basename(href))[0]
        name = _clean_unicode(a.get_text(" ", strip=True))

        thead = a.find_previous("thead")
        category = _clean_unicode(thead.get_text(" ", strip=True)) if thead else ""

        items.append(
            DatasetLink(
                id=series_id,
                provider="eia_psm",
                category=category,
                name=name,
                url=series_page_url,   # URL da página da série (não é o .xls ainda)
                filename="",           # será definido no download; manter vazio aqui
                ext="xls",
            )
        )

    # de-dup por id
    uniq = {}
    for d in items:
        uniq[d.id] = d
    return list(uniq.values())


def _eia_psm_find_xls(session: requests.Session, series_page_url: str, timeout: int) -> Optional[str]:
    """
    Dada a página .htm de uma série, encontra o href do 'Download Series History'
    (ou qualquer <a> que termine com .xls). Retorna URL absoluta do .xls.
    """
    r = session.get(series_page_url, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # 1) Preferência por texto 'Download Series History'
    link = soup.find("a", string=re.compile(r"Download Series History", re.I))
    if link and link.get("href"):
        return urljoin(series_page_url, link["href"])

    # 2) Primeiro <a> com .xls
    any_xls = soup.find("a", href=re.compile(r"\.xls$", re.I))
    if any_xls and any_xls.get("href"):
        return urljoin(series_page_url, any_xls["href"])

    # 3) Plano C: dentro do bloco #tlinks (quando existe)
    tlinks = soup.find(id="tlinks")
    if tlinks:
        xlsa = tlinks.find("a", href=re.compile(r"\.xls$", re.I))
        if xlsa and xlsa.get("href"):
            return urljoin(series_page_url, xlsa["href"])

    return None


def _eia_psm_list(cfg: dict,
                  session: Optional[requests.Session] = None,
                  logger: Optional[logging.Logger] = None) -> List[DatasetLink]:
    session = session or _requests_session(cfg)
    logger = logger or _setup_logger(cfg)
    logger.info("EIA_PSM: fetching index")
    r = session.get(_EIA_PSM_INDEX, timeout=cfg["scraper"]["timeout_s"])
    r.raise_for_status()
    html = r.text
    items = _eia_psm_parse_index(html)
    logger.info(f"EIA_PSM: parsed {len(items)} series pages")
    # Ordena por categoria + nome para estabilidade visual
    items.sort(key=lambda d: (d.category or "", d.name or ""))
    return items


def _eia_psm_download(cfg: dict, dataset_id: str, overwrite: bool = False,
                      session: Optional[requests.Session] = None,
                      logger: Optional[logging.Logger] = None) -> str:
    """
    dataset_id pode ser:
      - id da série (ex.: 'pet_sum_snd_d_r20_mbbl_m_cur')
      - URL para a página .htm da série
    Salva o .xls em data/raw/<arquivo.xls>.
    """
    session = session or _requests_session(cfg)
    logger = logger or _setup_logger(cfg)
    timeout = int(cfg["scraper"]["timeout_s"])

    # Resolver série -> URL da página .htm
    if dataset_id.lower().startswith("http"):
        series_page_url = dataset_id
        series_id = os.path.splitext(os.path.basename(urlparse(series_page_url).path))[0]
    else:
        all_items = _eia_psm_list(cfg, session=session, logger=logger)
        idx = {d.id: d for d in all_items}
        if dataset_id not in idx:
            # ajuda amigável
            suggestions = [k for k in idx if dataset_id.lower() in k.lower()]
            raise ValueError(
                f"EIA_PSM: id não encontrado: {dataset_id}. "
                f"Sugestões: {suggestions[:8]}"
            )
        series_page_url = idx[dataset_id].url
        series_id = dataset_id

    # Achar o .xls na página da série
    xls_url = _eia_psm_find_xls(session, series_page_url, timeout)
    if not xls_url:
        raise RuntimeError(f"EIA_PSM: nenhum link .xls encontrado em {series_page_url}")

    fname = _filename_from_url(xls_url) or f"{series_id}.xls"
    dest_dir = cfg["paths"]["data_raw"]
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, fname)

    if os.path.exists(dest_path) and not overwrite:
        logger.info(f"EIA_PSM: arquivo já existe (skip): {dest_path}")
        return dest_path

    logger.info(f"EIA_PSM: baixando {xls_url} -> {dest_path}")
    return _save_stream(session, xls_url, dest_path, timeout, logger)


def _eia_psm_resolve(cfg: dict, dataset_id: str,
                     session: Optional[requests.Session] = None,
                     logger: Optional[logging.Logger] = None) -> str:
    """
    Retorna a URL direta do XLS ('Download Series History') para uma série do PSM,
    sem salvar localmente.
    """
    session = session or _requests_session(cfg)
    logger = logger or _setup_logger(cfg)
    timeout = int(cfg["scraper"]["timeout_s"])

    # Resolver série -> URL da página .htm
    if dataset_id.lower().startswith("http"):
        series_page_url = dataset_id
    else:
        all_items = _eia_psm_list(cfg, session=session, logger=logger)
        idx = {d.id: d for d in all_items}
        if dataset_id not in idx:
            suggestions = [k for k in idx if dataset_id.lower() in k.lower()]
            raise ValueError(
                f"EIA_PSM: id não encontrado: {dataset_id}. "
                f"Sugestões: {suggestions[:8]}"
            )
        series_page_url = idx[dataset_id].url

    xls_url = _eia_psm_find_xls(session, series_page_url, timeout)
    if not xls_url:
        raise RuntimeError(f"EIA_PSM: nenhum link .xls encontrado em {series_page_url}")
    return xls_url


# ---------------------------
# PROVIDER: CONAB
# ---------------------------

_CONAB_BASE = "https://portaldeinformacoes.conab.gov.br"
_CONAB_PAGE = f"{_CONAB_BASE}/download-arquivos.html"


def _conab_parse_downloads(html: str) -> List[DatasetLink]:
    """
    Parse the CONAB downloads page and return all datasets with categories.
    We capture any anchor with href starting with /downloads/arquivos/.
    Category = nearest previous h4.card-title.
    Name     = anchor text (cleaned).
    """
    soup = BeautifulSoup(html, "html.parser")
    results: List[DatasetLink] = []

    # Strategy: find all h4.card-title, then gather links until the next h4
    h4s = soup.find_all("h4", class_="card-title")
    for h4 in h4s:
        category = _clean_unicode(h4.get_text(strip=True))
        # iterate through siblings until the next h4
        sib = h4.next_sibling
        while sib:
            # break at next h4
            if getattr(sib, "name", None) == "h4" and "card-title" in (sib.get("class") or []):
                break
            # look for anchors
            for a in getattr(sib, "find_all", lambda *a, **k: [])("a", href=True):
                href = a["href"]
                if isinstance(href, str) and href.startswith("/downloads/arquivos/"):
                    url = urljoin(_CONAB_BASE, href)
                    name_text = a.get_text(separator=" ", strip=True)
                    name_text = _clean_unicode(name_text.lstrip("- "))
                    filename = _filename_from_url(url)
                    ext = _ext(filename)
                    did = _slugify(f"conab__{category}__{name_text}__{filename}")
                    results.append(
                        DatasetLink(
                            id=did,
                            provider="conab",
                            category=category,
                            name=name_text,
                            url=url,
                            filename=filename,
                            ext=ext,
                        )
                    )
            sib = sib.next_sibling
    # de-dup just in case (same filename under slight DOM differences)
    uniq = {}
    for d in results:
        uniq[d.id] = d
    return list(uniq.values())


def _conab_list(cfg: dict, session: Optional[requests.Session] = None, logger: Optional[logging.Logger] = None) -> List[DatasetLink]:
    session = session or _requests_session(cfg)
    logger = logger or _setup_logger(cfg)
    logger.info("CONAB: fetching downloads page")
    r = session.get(_CONAB_PAGE, timeout=cfg["scraper"]["timeout_s"])
    r.raise_for_status()
    try:
        text = r.content.decode("utf-8")
    except UnicodeDecodeError:
        # fallback defensivo
        text = r.content.decode("latin-1", errors="replace")
    items = _conab_parse_downloads(text)
    logger.info(f"CONAB: parsed {len(items)} dataset links")
    return items


def _conab_download(cfg: dict, dataset_id: str, overwrite: bool = False,
                    session: Optional[requests.Session] = None,
                    logger: Optional[logging.Logger] = None) -> str:
    session = session or _requests_session(cfg)
    logger = logger or _setup_logger(cfg)
    all_items = _conab_list(cfg, session=session, logger=logger)
    idx = {d.id: d for d in all_items}
    if dataset_id not in idx:
        raise ValueError(f"Dataset id not found for CONAB: {dataset_id}")
    item = idx[dataset_id]
    dest_dir = cfg["paths"]["data_raw"]
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, item.filename)
    if os.path.exists(dest_path) and not overwrite:
        logger.info(f"CONAB: file already exists (skip): {dest_path}")
        return dest_path
    logger.info(f"CONAB: downloading {item.url} -> {dest_path}")
    return _save_stream(session, item.url, dest_path, cfg["scraper"]["timeout_s"], logger)


# ---------------------------
# provider registry (extensible)
# ---------------------------

_PROVIDERS = {
    "conab": {
        "list": _conab_list,
        "download": _conab_download,
    }
}

# ---------------------------
# provider registry (extensible)
# ---------------------------

_PROVIDERS.update({
    "eia_psm": {
        "list": _eia_psm_list,
        "download": _eia_psm_download,
        "resolve": _eia_psm_resolve,   # <— novo
    },
    "eia": {  # alias
        "list": _eia_psm_list,
        "download": _eia_psm_download,
        "resolve": _eia_psm_resolve,   # <— novo
    },
})



# ---------------------------
# public API
# ---------------------------

def resolve_dataset_url(provider: str, dataset_id: str, cfg: dict) -> str:
    """
    Retorna a URL direta do arquivo no provider (quando suportado).
    Para 'eia'/'eia_psm', retorna o link do XLS 'Download Series History'.
    """
    provider = provider.lower().strip()
    if provider not in _PROVIDERS or "resolve" not in _PROVIDERS[provider]:
        raise ValueError(f"Provider '{provider}' não suporta resolução de URL direta.")
    return _PROVIDERS[provider]["resolve"](cfg, dataset_id)


def list_datasets(provider: str, cfg: dict) -> List[dict]:
    """
    Return a list of dataset dicts for the given provider.
    Each dict: {id, provider, category, name, url, filename, ext}
    """
    provider = provider.lower().strip()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    items = _PROVIDERS[provider]["list"](cfg)
    return [d.to_dict() for d in items]


def download_dataset(provider: str, dataset_id: str, cfg: dict, overwrite: bool = False) -> str:
    """
    Download a dataset by id for the given provider; returns saved path.
    """
    provider = provider.lower().strip()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    return _PROVIDERS[provider]["download"](cfg, dataset_id, overwrite=overwrite)


def dump_index(provider: str, cfg: dict, out_json_path: Optional[str] = None) -> List[dict]:
    """
    Convenience: builds and (optionally) writes a JSON index of datasets.
    """
    items = list_datasets(provider, cfg)
    if out_json_path:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    return items
