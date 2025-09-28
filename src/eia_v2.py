# src/providers/eia_v2.py
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import requests
import pandas as pd

EIA_BASE = "https://api.eia.gov"
EIA_V2 = f"{EIA_BASE}/v2"


# ---------- models ----------

@dataclass
class EIAHTTP:
    url: str
    params: Dict[str, Any]
    status: int
    ok: bool
    json: Dict[str, Any]
    error: Optional[str] = None


# ---------- helpers ----------

def _route(*parts: str, data: bool = False) -> str:
    clean = [p.strip("/ ") for p in parts if p and p.strip("/ ")]
    base = "/".join([EIA_V2] + clean)
    return base + ("/data" if data else "")


def _encode_facets(
    facets: Optional[Mapping[str, Union[str, int, Iterable[Union[str, int]]]]]
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if not facets:
        return params
    for k, v in facets.items():
        key = f"facets[{k}][]"
        if isinstance(v, (list, tuple, set)):
            params[key] = [str(x) for x in v]
        else:
            params[key] = [str(v)]
    return params


def _encode_data_columns(columns: Optional[Iterable[str]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if columns:
        params["data[]"] = [str(c) for c in columns]
    return params


def _encode_sort(sort: Optional[List[Tuple[str, str]]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if not sort:
        return params
    for i, (col, direction) in enumerate(sort):
        params[f"sort[{i}][column]"] = str(col)
        params[f"sort[{i}][direction]"] = str(direction).lower()
    return params


# ---------- client ----------

class EIAClient:
    """
    Cliente minimalista para EIA API v2.1.
    - metadata(*route_parts)
    - facet_values(*route_parts, facet_id=...)
    - data(*route_parts, ...)
    - auto_data(*route_parts, ...) -> (pd.DataFrame, meta)
    - series_v1(series_id)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        cfg: Optional[dict] = None,
        timeout: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        # helper simples p/ ler cfg aninhado
        def _cfg_get(d: Optional[dict], path: tuple[str, ...], default=None):
            cur = d or {}
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur

        cfg = cfg or {}
        key_from_cfg = _cfg_get(cfg, ("api_keys", "eia"))
        self.api_key = (api_key or key_from_cfg or os.environ.get("EIA_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("EIA API key não informada (cfg.api_keys.eia ou env EIA_API_KEY).")

        self.timeout = timeout if timeout is not None else _cfg_get(cfg, ("scraper", "timeout_s"), 30)
        self.log = logger or logging.getLogger("eia")

        self.sess = requests.Session()
        self.sess.headers.update({
            "Accept": "application/json, */*;q=0.1",
            "Accept-Charset": "utf-8, iso-8859-1;q=0.8",
            "Accept-Language": "pt-BR, pt;q=0.9, en;q=0.8",
            "User-Agent": "FinDataCleaner/0.4 (EIA v2.1 client minimal)",
        })

    # --- HTTP base ---

    def _get(self, url: str, params: Dict[str, Any]) -> EIAHTTP:
        try:
            resp = self.sess.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as e:
            return EIAHTTP(url=url, params=params, status=0, ok=False, json={}, error=str(e))

        try:
            j = resp.json()
        except Exception:
            j = {}

        ok = resp.status_code == 200
        err = None
        if not ok:
            # tenta extrair msg útil do servidor
            err = (
                j.get("message")
                or j.get("error")
                or (j.get("response") or {}).get("error")
                or resp.text[:400]
            )
        return EIAHTTP(url=resp.url, params=params, status=resp.status_code, ok=ok, json=j, error=err)

    # --- discovery ---

    def metadata(self, *route_parts: str, extra: Optional[Dict[str, Any]] = None) -> EIAHTTP:
        url = _route(*route_parts, data=False)
        params: Dict[str, Any] = {"api_key": self.api_key}
        if extra:
            params.update(extra)
        return self._get(url, params)

    def facet_values(self, *route_parts: str, facet_id: str, extra: Optional[Dict[str, Any]] = None) -> EIAHTTP:
        url = _route(*route_parts, "facet", facet_id, data=False)
        params: Dict[str, Any] = {"api_key": self.api_key}
        if extra:
            params.update(extra)
        return self._get(url, params)

    # --- data ---

    def data(
        self,
        *route_parts: str,
        columns: Optional[Iterable[str]] = None,
        facets: Optional[Mapping[str, Union[str, int, Iterable[Union[str, int]]]]] = None,
        frequency: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        sort: Optional[List[Tuple[str, str]]] = None,
        offset: int = 0,
        length: int = 5000,
        extra: Optional[Dict[str, Any]] = None,
    ) -> EIAHTTP:
        url = _route(*route_parts, data=True)
        params: Dict[str, Any] = {"api_key": self.api_key, "offset": offset, "length": length}
        if frequency:
            params["frequency"] = frequency
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        params.update(_encode_data_columns(columns))
        params.update(_encode_facets(facets))
        params.update(_encode_sort(sort))
        if extra:
            params.update(extra)
        return self._get(url, params)

    def auto_data(
        self,
        *route_parts: str,
        columns: Optional[Iterable[str]] = None,
        facets: Optional[Mapping[str, Union[str, int, Iterable[Union[str, int]]]]] = None,
        frequency: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        sort: Optional[List[Tuple[str, str]]] = None,
        page_len: int = 5000,
        max_pages: Optional[int] = None,
        sleep_s: float = 0.1,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        frames: List[pd.DataFrame] = []
        offset = 0
        pages = 0
        meta: Dict[str, Any] = {}

        while True:
            res = self.data(
                *route_parts,
                columns=columns,
                facets=facets,
                frequency=frequency,
                start=start,
                end=end,
                sort=sort,
                offset=offset,
                length=page_len,
                extra=extra,
            )

            # se HTTP falhou, devolve meta com erro e para
            if not res.ok:
                meta = {"error": res.error, "status": res.status, "url": res.url}
                break

            rj = res.json or {}
            resp = rj.get("response", {})
            if not meta:
                for k in ("total", "dateFormat", "frequency", "warning", "description", "request"):
                    if k in resp:
                        meta[k] = resp[k]
                if "apiVersion" in rj:
                    meta["apiVersion"] = rj["apiVersion"]

            data = resp.get("data", [])
            if not data:
                # sem mais páginas
                break

            frames.append(pd.DataFrame(data))
            got = len(data)
            offset += got
            pages += 1

            if max_pages and pages >= max_pages:
                break
            if got < page_len:
                break

            time.sleep(sleep_s)  # respeitar throttling

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        # normaliza nomes
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
        return df, meta

    # --- legacy v1 ---

    def series_v1(self, series_id: str, extra: Optional[Dict[str, Any]] = None) -> EIAHTTP:
        url = _route("seriesid", series_id, data=False)
        params: Dict[str, Any] = {"api_key": self.api_key}
        if extra:
            params.update(extra)
        return self._get(url, params)


# ---------- opcional: índice a partir do OpenAPI YAML ----------

def index_from_openapi(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Constrói um índice (árvore rasa) de rotas a partir do OpenAPI YAML da EIA (spec['paths']).
    Útil para pré-popular o browser de rotas no Streamlit sem precisar “descobrir” online.

    Retorna algo como:
    {
      "electricity": {"retail-sales": {}, "state-electricity-profiles": {}, ...},
      "coal": {"mine-production": {}, "price-by-rank": {}, ...},
      ...
    }
    """
    tree: Dict[str, Any] = {}
    paths = spec.get("paths", {})
    for p in paths.keys():
        if not p.startswith("/v2/"):
            continue
        segs = [s for s in p.split("/") if s]
        # segs exemplo: ['v2','coal','mine-production','facet','{facet_id}'] ou ['v2','aeo','{route1}','data']
        if len(segs) < 2:
            continue
        root = segs[1]
        node = tree.setdefault(root, {})
        # adiciona apenas os primeiros níveis “semânticos” (antes de facet/data/param)
        current = node
        for s in segs[2:]:
            if s in ("data", "facet", "{facet_id}"):
                break
            current = current.setdefault(s, {})
    return tree
