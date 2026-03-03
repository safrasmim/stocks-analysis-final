"""Utilities for ingesting TDWL news/announcements from Mubasher mix2 endpoints."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import pandas as pd
import requests

MUBASHER_BASE_URL = "https://data-sa9.mubasher.net/mix2"
REQUEST_TIMEOUT_SECONDS = 30

REQUEST_TYPE_ANNOUNCEMENT = 25
REQUEST_TYPE_NEWS = 27

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MubasherRequestConfig:
    sid: str = "sid"
    uid: str = "123"
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

    formats = (
        "%Y%m%d%H%M%S",
        "%Y%m%d%H%M",
        "%Y%m%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
    )
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    try:
        parsed = pd.to_datetime(value, utc=True)
        if pd.isna(parsed):
            raise ValueError("datetime parse returned NaT")
        return parsed.to_pydatetime()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unsupported datetime format: {value}") from exc


def build_mix2_url(*, rt: int, config: MubasherRequestConfig, extra_params: dict[str, Any] | None = None) -> str:
    query = {
        "SID": config.sid,
        "UID": config.uid,
        "RT": rt,
        "E": config.exchange,
        "UE": config.user_exchange,
        "L": config.language,
        "AE": config.include_archived_events,
        "UNC": config.uncached,
        "M": config.mode,
        "H": config.include_header,
    }
    if extra_params:
        query.update(extra_params)
    return f"{MUBASHER_BASE_URL}?{urlencode(query)}"


def fetch_mix2_json(url: str, *, session: requests.Session | None = None) -> dict[str, Any]:
    client = session or requests.Session()
    response = client.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("mix2 response is not a JSON object")
    return payload


def _header_index_map(header: Iterable[str], required: Iterable[str]) -> dict[str, int]:
    header_list = list(header)
    idx: dict[str, int] = {}
    for key in required:
        if key not in header_list:
            raise ValueError(f"Required header '{key}' missing in mix2 response")
        idx[key] = header_list.index(key)
    return idx


def parse_mix2_items(payload: dict[str, Any], *, item_type: str, allowed_tickers: set[str] | None = None) -> list[ParsedItem]:
    header = payload.get("HED", {}).get("NWSL")
    records = payload.get("DAT", {}).get("NWSL", [])

    if not isinstance(header, list):
        raise ValueError("Invalid mix2 payload: HED.NWSL missing or malformed")
    if not isinstance(records, list):
        raise ValueError("Invalid mix2 payload: DAT.NWSL missing or malformed")

    idx = _header_index_map(header, ["ID", "E", "S", "HED", "DT", "L", "PRV"])
    rows: list[ParsedItem] = []

    for raw_record in records:
        if not isinstance(raw_record, str):
            LOGGER.warning("Skipping non-string record in DAT.NWSL: %r", raw_record)
            continue

        parts = raw_record.split("|")
        max_index = max(idx.values())
        if len(parts) <= max_index:
            LOGGER.warning("Skipping short record with %s fields: %s", len(parts), raw_record)
            continue

        ticker = parts[idx["S"]].strip()
        if allowed_tickers and ticker not in allowed_tickers:
            continue

        try:
            published_at = _parse_datetime(parts[idx["DT"]])
        except ValueError:
            LOGGER.warning("Skipping record with invalid DT value: %s", parts[idx["DT"]])
            continue

        rows.append(
            ParsedItem(
                item_id=parts[idx["ID"]].strip(),
                item_type=item_type,
                exchange=parts[idx["E"]].strip(),
                ticker=ticker,
                provider=parts[idx["PRV"]].strip(),
                language=parts[idx["L"]].strip(),
                headline=parts[idx["HED"]].strip(),
                published_at=published_at,
                raw_datetime=parts[idx["DT"]].strip(),
            )
        )

    return rows


def _load_existing(store_path: Path) -> pd.DataFrame:
    if not store_path.exists():
        return pd.DataFrame(
            columns=[
                "item_id",
                "item_type",
                "exchange",
                "ticker",
                "provider",
                "language",
                "headline",
                "published_at",
                "raw_datetime",
                "fetched_at",
            ]
        )

    if store_path.suffix == ".parquet":
        frame = pd.read_parquet(store_path)
    else:
        frame = pd.read_csv(store_path)

    if "published_at" in frame.columns:
        frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    return frame


def upsert_incremental_items(new_items: list[ParsedItem], *, store_path: Path) -> pd.DataFrame:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(store_path)

    if not new_items:
        return existing

    incoming = pd.DataFrame(
        [
            {
                "item_id": item.item_id,
                "item_type": item.item_type,
                "exchange": item.exchange,
                "ticker": item.ticker,
                "provider": item.provider,
                "language": item.language,
                "headline": item.headline,
                "published_at": item.published_at,
                "raw_datetime": item.raw_datetime,
                "fetched_at": datetime.now(timezone.utc),
            }
            for item in new_items
        ]
    )

    combined = pd.concat([existing, incoming], ignore_index=True)
    combined["published_at"] = pd.to_datetime(combined["published_at"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["published_at", "item_id", "ticker"])
    combined = combined.sort_values(["published_at", "item_id"]).drop_duplicates(
        subset=["item_id", "item_type", "ticker"], keep="last"
    )

    if store_path.suffix == ".parquet":
        combined.to_parquet(store_path, index=False)
    else:
        combined.to_csv(store_path, index=False)
    return combined


def compute_latest_cursor(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty:
        return {}

    grouped = (
        frame.groupby(["item_type", "ticker"], as_index=False)["published_at"]
        .max()
        .dropna()
    )

    return {
        f"{row.item_type}:{row.ticker}": pd.Timestamp(row.published_at).isoformat()
        for row in grouped.itertuples(index=False)
    }


def save_cursor(cursor: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cursor, indent=2, sort_keys=True), encoding="utf-8")


def load_cursor(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Cursor file is corrupted (%s). Starting with an empty cursor.", path)
        return {}
    return raw if isinstance(raw, dict) else {}


def filter_items_newer_than_cursor(items: list[ParsedItem], *, cursor: dict[str, str]) -> list[ParsedItem]:
    kept: list[ParsedItem] = []
    for item in items:
        cursor_key = f"{item.item_type}:{item.ticker}"
        cursor_value = cursor.get(cursor_key)
        if not cursor_value:
            kept.append(item)
            continue

        try:
            cursor_dt = pd.to_datetime(cursor_value, utc=True).to_pydatetime()
        except Exception:  # noqa: BLE001
            kept.append(item)
            continue

        if item.published_at > cursor_dt:
            kept.append(item)
    return kept
