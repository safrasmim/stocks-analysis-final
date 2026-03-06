"""
mubasher_tdwl.py  —  UPGRADED
Robust TDWL news/announcements ingestion from Mubasher mix2.
"""
from __future__ import annotations

import time
import logging
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlencode

import pandas as pd
import requests

MUBASHER_BASE_URL         = "https://data-sa9.mubasher.net/mix2"
REQUEST_TIMEOUT_SECONDS   = 30
REQUEST_TYPE_ANNOUNCEMENT = 25
REQUEST_TYPE_NEWS         = 27
API_RECORD_LIMIT          = 1000
NEWS_CHUNK_DAYS           = 7
ANN_CHUNK_DAYS            = 30
MAX_RETRIES               = 3
DATE_FMT                  = "%Y%m%d%H%M%S"

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MubasherRequestConfig:
    sid: str  = "sid"
    uid: str  = "123"
    exchange: str = "TDWL"
    user_exchange: str = "TDWL"
    language: str = "EN"
    include_archived_events: int = 1
    uncached: int = 0
    mode: int = 1
    include_header: int = 1


@dataclass(frozen=True)
class ParsedItem:
    item_id: str
    item_type: str
    exchange: str
    ticker: str
    provider: str
    language: str
    headline: str
    published_at: datetime
    raw_datetime: str


def _parse_datetime(raw: str) -> datetime:
    value = (raw or "").strip()
    if not value:
        raise ValueError("Empty datetime value.")
    for fmt in (
        "%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S", "%d/%m/%Y",
    ):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = pd.to_datetime(value, utc=True)
        if pd.isna(parsed):
            raise ValueError("NaT")
        return parsed.to_pydatetime()
    except Exception as exc:
        raise ValueError(f"Unsupported datetime format: {value}") from exc


def build_mix2_url(
    *,
    rt: int,
    config: MubasherRequestConfig,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    extra_params: Optional[dict] = None,
) -> str:
    query: dict = {
        "SID": config.sid, "UID": config.uid, "RT": rt,
        "E": config.exchange, "UE": config.user_exchange,
        "L": config.language, "AE": config.include_archived_events,
        "UNC": config.uncached, "M": config.mode, "H": config.include_header,
    }
    if start_dt:
        query["SD"] = start_dt.strftime(DATE_FMT)
    if end_dt:
        query["ED"] = end_dt.strftime(DATE_FMT)
    if extra_params:
        query.update(extra_params)
    return f"{MUBASHER_BASE_URL}?{urlencode(query)}"


def fetch_mix2_json(url: str, *, session=None) -> dict:
    client = session or requests.Session()
    last_exc = RuntimeError("No attempts made")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("mix2 response is not a JSON object")
            record_count = len((payload.get("DAT") or {}).get("NWSL") or [])
            if record_count >= API_RECORD_LIMIT:
                LOGGER.warning("Hit %d-record API limit — narrow date window. URL: %s", API_RECORD_LIMIT, url[:120])
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            wait = 2 ** attempt
            LOGGER.warning("Attempt %d/%d failed (%s). Retrying in %ds...", attempt, MAX_RETRIES, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"All {MAX_RETRIES} fetch attempts failed: {last_exc}") from last_exc


def _header_index_map(header: Iterable, required: Iterable) -> dict:
    header_list = list(header)
    idx = {}
    for key in required:
        if key not in header_list:
            raise ValueError(f"Required header '{key}' missing in mix2 response")
        idx[key] = header_list.index(key)
    return idx


def parse_mix2_items(payload: dict, *, item_type: str, allowed_tickers=None) -> list:
    """Parse a mix2 API response into ParsedItem records.

    Supports both response types:
      RT=27  ->  HED/DAT keyed as NWSL
      RT=25  ->  HED/DAT keyed as ANNL
    """
    hed_section = payload.get("HED", {})
    dat_section = payload.get("DAT", {})

    if not isinstance(hed_section, dict) or not isinstance(dat_section, dict):
        raise ValueError("Invalid mix2 payload: HED or DAT is not a dict")

    # Auto-detect the list key present in this response
    data_key = next((k for k in ("NWSL", "ANNL") if k in hed_section), None)
    if not data_key:
        raise ValueError(
            f"Unknown payload structure — no NWSL or ANNL key found. "
            f"HED keys present: {list(hed_section.keys())}"
        )

    header_raw = hed_section.get(data_key)
    records    = dat_section.get(data_key, [])

    # Header is a pipe-delimited STRING e.g. "ID|DT|E|S|HED|..."
    if not isinstance(header_raw, str) or not header_raw.strip():
        raise ValueError(f"HED.{data_key} is not a non-empty string header")
    if not isinstance(records, list):
        raise ValueError(f"DAT.{data_key} is not a list")

    header = header_raw.split("|")

    # Helper: column index with optional fallbacks, returns None if absent
    def col(*names):
        for name in names:
            if name in header:
                return header.index(name)
        return None

    idx_id  = col("ID")
    idx_dt  = col("DT")
    idx_e   = col("E")
    idx_s   = col("S")
    idx_hed = col("HED")
    idx_l   = col("L")
    # PRV absent in both RT=25/27; fall back to SRC (announcements) or CREUSR (news)
    idx_prv = col("PRV", "SRC", "CREUSR")

    # Abort early if essential columns are missing
    missing = [name for name, idx in [("ID", idx_id), ("DT", idx_dt),
                                       ("S",  idx_s),  ("HED", idx_hed)]
               if idx is None]
    if missing:
        raise ValueError(
            f"Required columns {missing} missing from {data_key} header. "
            f"Full header: {header_raw[:200]}"
        )

    # Highest index we'll ever access
    max_required = max(v for v in [idx_id, idx_dt, idx_e, idx_s, idx_hed, idx_l, idx_prv]
                       if v is not None)

    rows = []
    for raw_record in records:
        if not isinstance(raw_record, str):
            LOGGER.warning("Skipping non-string record: %r", raw_record)
            continue
        parts = raw_record.split("|")
        if len(parts) <= max_required:
            LOGGER.warning("Skipping short record (%d fields): %s", len(parts), raw_record[:80])
            continue

        # Ticker can be comma-separated "4240,4240.B" — use primary only
        ticker_raw = parts[idx_s].strip()
        ticker     = ticker_raw.split(",")[0].strip()

        if allowed_tickers and ticker not in allowed_tickers:
            continue

        try:
            published_at = _parse_datetime(parts[idx_dt])
        except ValueError:
            LOGGER.warning("Skipping bad DT value: %s", parts[idx_dt])
            continue

        def safe(idx):
            """Return stripped part at idx, empty string if idx is None or out of bounds."""
            if idx is None or idx >= len(parts):
                return ""
            return parts[idx].strip()

        rows.append(ParsedItem(
            item_id      = safe(idx_id),
            item_type    = item_type,
            exchange     = safe(idx_e) or "TDWL",
            ticker       = ticker,
            provider     = safe(idx_prv),
            language     = safe(idx_l) or "EN",
            headline     = safe(idx_hed),
            published_at = published_at,
            raw_datetime = parts[idx_dt].strip(),
        ))

    return rows


def fetch_date_range(*, rt, config, start_dt, end_dt, item_type, allowed_tickers=None, session=None, chunk_days=NEWS_CHUNK_DAYS):
    """Fetch a full date range split into chunk_days windows to avoid 1000-record truncation."""
    all_items = []
    cursor_dt = start_dt
    while cursor_dt < end_dt:
        window_end = min(cursor_dt + timedelta(days=chunk_days), end_dt)
        url = build_mix2_url(rt=rt, config=config, start_dt=cursor_dt, end_dt=window_end)
        LOGGER.info("Fetching %s  %s -> %s", item_type, cursor_dt.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d"))
        try:
            payload = fetch_mix2_json(url, session=session)
            items   = parse_mix2_items(payload, item_type=item_type, allowed_tickers=allowed_tickers)
            all_items.extend(items)
            LOGGER.info("  -> %d items", len(items))
        except Exception as exc:
            LOGGER.error("Window fetch failed (%s -> %s): %s", cursor_dt.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d"), exc)
        cursor_dt = window_end
    LOGGER.info("fetch_date_range: total %d %s items", len(all_items), item_type)
    return all_items


def fetch_latest(*, rt, config, item_type, allowed_tickers=None, session=None, chunk_days=NEWS_CHUNK_DAYS):
    """Fetch only the last chunk_days window (real-time / incremental)."""
    now   = datetime.now(timezone.utc)
    start = now - timedelta(days=chunk_days)
    return fetch_date_range(rt=rt, config=config, start_dt=start, end_dt=now,
                            item_type=item_type, allowed_tickers=allowed_tickers,
                            session=session, chunk_days=chunk_days)


def _load_existing(store_path: Path) -> pd.DataFrame:
    if not store_path.exists():
        return pd.DataFrame(columns=["item_id","item_type","exchange","ticker","provider",
                                     "language","headline","published_at","raw_datetime","fetched_at"])
    frame = pd.read_parquet(store_path) if store_path.suffix == ".parquet" else pd.read_csv(store_path)
    if "published_at" in frame.columns:
        frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    return frame


def upsert_incremental_items(new_items: list, *, store_path: Path) -> pd.DataFrame:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(store_path)
    if not new_items:
        return existing
    incoming = pd.DataFrame([{
        "item_id": item.item_id, "item_type": item.item_type, "exchange": item.exchange,
        "ticker": item.ticker, "provider": item.provider, "language": item.language,
        "headline": item.headline, "published_at": item.published_at,
        "raw_datetime": item.raw_datetime, "fetched_at": datetime.now(timezone.utc),
    } for item in new_items])
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined["published_at"] = pd.to_datetime(combined["published_at"], utc=True, errors="coerce")
    combined = (combined.dropna(subset=["published_at","item_id","ticker"])
                        .sort_values(["published_at","item_id"])
                        .drop_duplicates(subset=["item_id","item_type","ticker"], keep="last"))
    if store_path.suffix == ".parquet":
        combined.to_parquet(store_path, index=False)
    else:
        combined.to_csv(store_path, index=False)
    LOGGER.info("Upserted %d new -> %d total in %s", len(new_items), len(combined), store_path)
    return combined


def compute_latest_cursor(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {}
    grouped = frame.groupby(["item_type","ticker"], as_index=False)["published_at"].max().dropna()
    return {f"{row.item_type}:{row.ticker}": pd.Timestamp(row.published_at).isoformat()
            for row in grouped.itertuples(index=False)}


def save_cursor(cursor: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cursor, indent=2, sort_keys=True), encoding="utf-8")


def load_cursor(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Cursor file corrupted (%s). Starting fresh.", path)
        return {}
    return raw if isinstance(raw, dict) else {}


def filter_items_newer_than_cursor(items: list, *, cursor: dict) -> list:
    kept = []
    for item in items:
        cursor_key   = f"{item.item_type}:{item.ticker}"
        cursor_value = cursor.get(cursor_key)
        if not cursor_value:
            kept.append(item)
            continue
        try:
            cursor_dt = pd.to_datetime(cursor_value, utc=True).to_pydatetime()
        except Exception:
            kept.append(item)
            continue
        if item.published_at > cursor_dt:
            kept.append(item)
    return kept
