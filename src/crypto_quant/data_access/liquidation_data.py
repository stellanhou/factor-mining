"""Auditable CryptoHFTData liquidation ingestion and causal hourly alignment.

Raw provider objects are kept byte-for-byte under ``market_data/raw``.  The
derived alignment is disposable and can always be rebuilt from those objects,
the daily object indexes, and :class:`~crypto_quant.data_access.market_data.MarketDataStore`.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import tarfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd
import requests

from crypto_quant.features.factors import FactorEngine
from crypto_quant.data_access.market_data import MarketDataStore, SPOT, USD_M_PERPETUAL, resolve_market_symbols


PROVIDER = "cryptohftdata"
EXCHANGE = "binance_futures"
DATA_TYPE = "liquidations"
PROVIDER_HISTORY_START = pd.Timestamp("2025-06-28T00:00:00Z")
SYMBOLS_URL = "https://api.cryptohftdata.com/v1/symbols"
# The provider documents S3 credential exchange on the compatibility endpoint;
# the v1-prefixed endpoint currently does not expose this operation.
S3_CREDENTIALS_URL = "https://api.cryptohftdata.com/s3-credentials"
COMPANION_TYPES = frozenset(
    {"trades", "mark_price", "open_interest", "orderbook", "ticker"}
)
KNOWN_DATA_TYPES = tuple(
    sorted((*COMPANION_TYPES, DATA_TYPE), key=len, reverse=True)
)
RAW_MANIFEST = "_download_manifest.jsonl"
UNIVERSE_MANIFEST = "_universe.json"
INDEX_DIRECTORY = "_object_index"


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _iso(value: object) -> str:
    return _utc(value).isoformat()


def _json_default(value: object) -> object:
    if isinstance(value, (datetime, pd.Timestamp)):
        return _iso(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_json_gzip_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, default=_json_default)
        handle.write("\n")
    os.replace(temporary, path)


def _read_json_gzip(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class LiquidationUniverseRecord:
    symbol: str
    start: str
    end: str
    spot_start: str
    spot_end: str
    futures_start: str
    futures_end: str
    funding_start: str
    funding_end: str
    oi_start: str
    oi_end: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def fetch_liquidation_symbols(
    session: Optional[requests.Session] = None,
) -> Set[str]:
    """Return the provider's current Binance-futures liquidation registry."""
    client = session or requests.Session()
    response = client.get(
        SYMBOLS_URL,
        params={"exchange": EXCHANGE, "data_type": DATA_TYPE},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        raise ValueError("CryptoHFTData symbols response has no symbols list")
    return {str(symbol).upper() for symbol in symbols}


def build_liquidation_universe(
    store: MarketDataStore,
    provider_symbols: Iterable[str],
    provider_start: pd.Timestamp = PROVIDER_HISTORY_START,
) -> Dict[str, Any]:
    """Intersect mapped local 1h spot/perpetual, funding, OI, and provider data."""
    spot = {
        row["symbol"]: row
        for row in store.spot_snapshot()
        if row["interval"] == "1h"
    }
    funding = {row["symbol"]: row for row in store.funding_snapshot()}
    oi = {
        row["symbol"]: row
        for row in store.open_interest_snapshot()
        if row.get("period") == "5m"
    }
    futures = {
        row["symbol"]: row
        for row in store.futures_price_snapshot("trade")
        if row["interval"] == "1h"
    }
    local_symbols = sorted(
        symbol for symbol in set(futures) & set(funding) & set(oi)
        if resolve_market_symbols(symbol).spot in spot
    )
    provider_set = {str(symbol).upper() for symbol in provider_symbols}

    eligible: List[LiquidationUniverseRecord] = []
    no_time_overlap: List[str] = []
    provider_missing = sorted(set(local_symbols) - provider_set)
    for symbol in sorted(set(local_symbols) & provider_set):
        spot_symbol = resolve_market_symbols(symbol).spot
        start = max(
            provider_start,
            _utc(spot[spot_symbol]["start"]),
            _utc(futures[symbol]["start"]),
            _utc(funding[symbol]["start"]),
            _utc(oi[symbol]["start"]),
        )
        end = min(
            _utc(spot[spot_symbol]["end"]),
            _utc(futures[symbol]["end"]),
            _utc(funding[symbol]["end"]),
            _utc(oi[symbol]["end"]),
        )
        if start > end:
            no_time_overlap.append(symbol)
            continue
        eligible.append(
            LiquidationUniverseRecord(
                symbol=symbol,
                start=_iso(start),
                end=_iso(end),
                spot_start=str(spot[spot_symbol]["start"]),
                spot_end=str(spot[spot_symbol]["end"]),
                futures_start=str(futures[symbol]["start"]),
                futures_end=str(futures[symbol]["end"]),
                funding_start=str(funding[symbol]["start"]),
                funding_end=str(funding[symbol]["end"]),
                oi_start=str(oi[symbol]["start"]),
                oi_end=str(oi[symbol]["end"]),
            )
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": PROVIDER,
        "exchange": EXCHANGE,
        "data_type": DATA_TYPE,
        "provider_history_start": _iso(provider_start),
        "local_counts": {
            "spot_1h": len(spot),
            "futures_trade_1h": len(futures),
            "funding": len(funding),
            "oi_5m": len(oi),
            "four_way_intersection": len(local_symbols),
        },
        "provider_symbol_count": len(provider_set),
        "eligible_count": len(eligible),
        "provider_missing_count": len(provider_missing),
        "provider_missing": provider_missing,
        "no_time_overlap_count": len(no_time_overlap),
        "no_time_overlap": no_time_overlap,
        "eligible": [
            {**record.to_dict(), "market_symbols": asdict(resolve_market_symbols(record.symbol))}
            for record in eligible
        ],
    }


def prepare_liquidation_universe(db_path: Path, raw_root: Path) -> Dict[str, Any]:
    """Build and persist the deterministic local/provider overlap manifest."""
    destination = Path(raw_root).resolve() / PROVIDER / UNIVERSE_MANIFEST
    universe = build_liquidation_universe(
        MarketDataStore(Path(db_path)), fetch_liquidation_symbols()
    )
    _write_json_atomic(destination, universe)
    return {**universe, "manifest": str(destination)}


class CryptoHFTS3Provider:
    """Refreshable read-only S3 client created from a CryptoHFTData API key."""

    def __init__(self, api_key: str, max_pool_connections: int = 32) -> None:
        if not api_key:
            raise ValueError("CryptoHFTData API key is required for bulk listing")
        self.api_key = api_key
        self.max_pool_connections = max(10, int(max_pool_connections))
        self._lock = threading.Lock()
        self._client = None
        self._bucket: Optional[str] = None
        self._expires_at: Optional[datetime] = None

    @property
    def bucket(self) -> str:
        self.client()
        if not self._bucket:
            raise RuntimeError("S3 credentials did not provide a bucket")
        return self._bucket

    def client(self):
        with self._lock:
            now = datetime.now(timezone.utc)
            if (
                self._client is None
                or self._expires_at is None
                or now >= self._expires_at - timedelta(minutes=5)
            ):
                self._refresh()
            return self._client

    def _refresh(self) -> None:
        try:
            from cryptohftdata.s3lite import S3LiteClient
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "bulk liquidation downloads require the 'liquidation-data' extra"
            ) from exc
        response = requests.post(
            S3_CREDENTIALS_URL,
            headers={"X-API-Key": self.api_key},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        credentials = payload.get("credentials") or {}
        required = ("access_key_id", "secret_access_key", "session_token")
        missing = [name for name in required if not credentials.get(name)]
        if missing or not payload.get("bucket") or not payload.get("endpoint"):
            raise ValueError("incomplete S3 credential response")
        expiry = payload.get("expires_at")
        if expiry:
            self._expires_at = _utc(expiry).to_pydatetime()
        else:
            self._expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=int(payload.get("expires_in", 3600))
            )
        self._bucket = str(payload["bucket"])
        self._client = S3LiteClient(
            endpoint=str(payload["endpoint"]),
            region=str(payload.get("region") or "auto"),
            access_key_id=credentials["access_key_id"],
            secret_access_key=credentials["secret_access_key"],
            session_token=credentials["session_token"],
            max_pool_connections=self.max_pool_connections,
            timeout=60,
            max_attempts=5,
        )


def parse_object_key(key: str) -> Optional[Dict[str, str]]:
    """Parse one provider key without truncating symbols containing underscores."""
    parts = key.split("/")
    if len(parts) != 4 or parts[0] != EXCHANGE:
        return None
    filename = parts[3]
    extension = next(
        (suffix for suffix in (".parquet.zst", ".parquet") if filename.endswith(suffix)),
        None,
    )
    if extension is None:
        return None
    stem = filename[: -len(extension)]
    for data_type in KNOWN_DATA_TYPES:
        suffix = "_" + data_type
        if stem.endswith(suffix):
            symbol = stem[: -len(suffix)]
            if symbol:
                return {
                    "exchange": parts[0],
                    "date": parts[1],
                    "hour": parts[2],
                    "symbol": symbol,
                    "data_type": data_type,
                    "extension": extension,
                }
    return None


def _list_hour(
    provider: CryptoHFTS3Provider,
    date: str,
    hour: int,
    active_symbols: Set[str],
) -> Dict[str, Any]:
    prefix = f"{EXCHANGE}/{date}/{hour:02d}/"
    client = provider.client()
    paginator = client.get_paginator("list_objects_v2")
    companion: Dict[str, Set[str]] = {}
    liquidations: List[Dict[str, Any]] = []
    for page in paginator.paginate(Bucket=provider.bucket, Prefix=prefix):
        for item in page.get("Contents", []) or []:
            key = str(item.get("Key", ""))
            parsed = parse_object_key(key)
            if parsed is None or parsed["symbol"] not in active_symbols:
                continue
            symbol = parsed["symbol"]
            if parsed["data_type"] in COMPANION_TYPES:
                companion.setdefault(symbol, set()).add(parsed["data_type"])
            elif parsed["data_type"] == DATA_TYPE:
                liquidations.append(
                    {
                        **parsed,
                        "key": key,
                        "size": int(item.get("Size", 0)),
                        "etag": str(item.get("ETag", "")).strip('"'),
                        "last_modified": item.get("LastModified"),
                    }
                )
    return {
        "hour": f"{hour:02d}",
        "companion": {
            symbol: sorted(types) for symbol, types in sorted(companion.items())
        },
        "liquidations": sorted(liquidations, key=lambda row: row["key"]),
    }


def _active_symbols_for_date(
    records: Sequence[Mapping[str, str]], date: str
) -> Set[str]:
    day_start = pd.Timestamp(date, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return {
        str(record["symbol"])
        for record in records
        if _utc(record["start"]) <= day_end and _utc(record["end"]) >= day_start
    }


def _record_windows(
    records: Sequence[Mapping[str, str]],
) -> Dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    return {
        str(record["symbol"]): (
            _utc(record["start"]).floor("h"),
            _utc(record["end"]).floor("h"),
        )
        for record in records
    }


def _active_symbols_from_windows(
    windows: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
    date: str,
    hour: Optional[int] = None,
) -> Set[str]:
    start = pd.Timestamp(
        f"{date}T{hour:02d}:00:00Z" if hour is not None else f"{date}T00:00:00Z"
    )
    end = start if hour is not None else start + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return {
        symbol
        for symbol, (window_start, window_end) in windows.items()
        if window_start <= end and window_end >= start
    }


def _active_symbols_for_hour(
    records: Sequence[Mapping[str, str]], date: str, hour: int
) -> Set[str]:
    timestamp = pd.Timestamp(f"{date}T{hour:02d}:00:00Z")
    return {
        str(record["symbol"])
        for record in records
        if _utc(record["start"]).floor("h") <= timestamp
        and _utc(record["end"]).floor("h") >= timestamp
    }


def _fetch_one(
    provider: CryptoHFTS3Provider,
    item: Mapping[str, Any],
) -> tuple[Dict[str, Any], bytes, List[Dict[str, Any]]]:
    key = str(item["key"])
    expected = int(item.get("size", 0))
    response = provider.client().get_object(Bucket=provider.bucket, Key=key)
    chunks: List[bytes] = []
    body = response["Body"]
    try:
        while True:
            chunk = body.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        body.close()
    payload = b"".join(chunks)
    written = len(payload)
    if expected and written != expected:
        raise IOError(f"short read for {key}: {written} of {expected} bytes")
    entry = {
        **dict(item),
        "member": key,
        "bytes": written,
    }
    aggregate_rows, invalid_notional_count = _aggregate_liquidation_payload(
        payload,
        symbol=str(item["symbol"]),
        source_key=key,
    )
    if invalid_notional_count:
        entry["invalid_notional_count"] = invalid_notional_count
    return entry, payload, aggregate_rows


def _write_hourly_aggregate(
    path: Path, aggregate_rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(aggregate_rows))
    if not frame.empty:
        frame = frame.sort_values(["timestamp", "symbol", "source_key"])
    temporary = path.with_name(path.name + ".part")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)
    return {
        "hourly_aggregate": str(path),
        "hourly_aggregate_rows": len(frame),
        "hourly_aggregate_bytes": path.stat().st_size,
    }


def _existing_hourly_aggregate_metadata(path: Path) -> Dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "liquidation audit requires the 'liquidation-data' extra"
        ) from exc
    return {
        "hourly_aggregate": str(path),
        "hourly_aggregate_rows": parquet.ParquetFile(path).metadata.num_rows,
        "hourly_aggregate_bytes": path.stat().st_size,
    }


def _materialize_archive_aggregate(
    archive: Path,
    members: Sequence[Mapping[str, Any]],
    aggregate_path: Path,
) -> Dict[str, Any]:
    aggregate_rows: List[Dict[str, Any]] = []
    with tarfile.open(archive, "r") as bundle:
        for entry in members:
            key = str(entry["key"])
            member = bundle.extractfile(key)
            if member is None:
                raise IOError(f"archive member is missing: {archive}::{key}")
            rows, invalid_notional_count = _aggregate_liquidation_payload(
                member.read(),
                symbol=str(entry["symbol"]),
                source_key=key,
            )
            aggregate_rows.extend(rows)
            if invalid_notional_count and isinstance(entry, dict):
                entry["invalid_notional_count"] = invalid_notional_count
    return _write_hourly_aggregate(aggregate_path, aggregate_rows)


def _archive_day(
    provider: CryptoHFTS3Provider,
    raw_root: Path,
    date: str,
    objects: Sequence[Mapping[str, Any]],
    workers: int,
) -> Dict[str, Any]:
    """Download one day into a deterministic tar while preserving member bytes."""
    archive = raw_root / "_daily_archives" / f"{date}.tar"
    manifest = raw_root / "_daily_manifests" / f"{date}.json"
    aggregate_path = raw_root / "_normalized_hourly" / f"{date}.parquet"
    if archive.is_file() and manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if int(payload.get("archive_bytes", -1)) == archive.stat().st_size:
            if not aggregate_path.is_file():
                payload.update(
                    _materialize_archive_aggregate(
                        archive, payload.get("members", []), aggregate_path
                    )
                )
                _write_json_atomic(manifest, payload)
            elif (
                "hourly_aggregate_rows" not in payload
                or "hourly_aggregate_bytes" not in payload
            ):
                payload.update(_existing_hourly_aggregate_metadata(aggregate_path))
                _write_json_atomic(manifest, payload)
            return {**payload, "status": "existing"}

    fetched: Dict[str, tuple[Dict[str, Any], bytes, List[Dict[str, Any]]]] = {}
    failures: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_one, provider, item): item for item in objects
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                entry, data, aggregate_rows = future.result()
            except Exception as exc:
                failures.append(
                    {"key": str(item["key"]), "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            fetched[str(entry["key"])] = (entry, data, aggregate_rows)
    if failures:
        raise IOError(
            f"failed to download {len(failures)} liquidation objects for {date}; "
            f"first={failures[0]}"
        )

    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".part")
    member_entries: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    with tarfile.open(temporary, "w") as bundle:
        for key in sorted(fetched):
            entry, data, rows = fetched[key]
            info = tarfile.TarInfo(name=key)
            info.size = len(data)
            modified = entry.get("last_modified")
            info.mtime = int(_utc(modified).timestamp()) if modified else 0
            info.mode = 0o444
            bundle.addfile(info, io.BytesIO(data))
            member_entries.append(entry)
            aggregate_rows.extend(rows)
    os.replace(temporary, archive)
    result = {
        "provider": PROVIDER,
        "exchange": EXCHANGE,
        "date": date,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "members": member_entries,
        "member_count": len(member_entries),
        "member_bytes": sum(int(row["bytes"]) for row in member_entries),
        "status": "downloaded",
    }
    result.update(_write_hourly_aggregate(aggregate_path, aggregate_rows))
    _write_json_atomic(manifest, result)
    return result


def _write_daily_coverage(
    raw_root: Path,
    day_index: Mapping[str, Any],
    windows: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> Path:
    date = str(day_index["date"])
    destination = raw_root / "_coverage_hourly" / f"{date}.parquet"
    if destination.is_file():
        return destination
    rows: List[Dict[str, Any]] = []
    for hour in day_index.get("hours", []):
        hour_number = int(hour["hour"])
        timestamp = pd.Timestamp(f"{date}T{hour_number:02d}:00:00Z")
        liquidation_by_symbol: Dict[str, int] = {}
        for item in hour.get("liquidations", []):
            symbol = str(item["symbol"])
            liquidation_by_symbol[symbol] = liquidation_by_symbol.get(symbol, 0) + 1
        companions = hour.get("companion", {})
        for symbol in sorted(
            _active_symbols_from_windows(windows, date, hour_number)
        ):
            file_count = liquidation_by_symbol.get(symbol, 0)
            companion_types = companions.get(symbol, [])
            if file_count:
                status = "valid_event"
            elif companion_types:
                status = "valid_zero"
            else:
                status = "source_unknown"
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "liquidation_coverage_status": status,
                    "liquidation_source_file_count": file_count,
                    "liquidation_companion_types": ",".join(companion_types),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["timestamp", "symbol"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, destination)
    return destination


def download_liquidation_history(
    db_path: Path,
    raw_root: Path,
    api_key: str,
    *,
    workers: int = 16,
    authorize: bool = False,
    start_date: Optional[object] = None,
    end_date: Optional[object] = None,
) -> Dict[str, Any]:
    """Download every existing liquidation object in the exact local overlap."""
    if not authorize:
        raise PermissionError("explicit public-download authorization is required")
    if workers < 1:
        raise ValueError("workers must be positive")

    raw_root = Path(raw_root).resolve() / PROVIDER
    store = MarketDataStore(Path(db_path))
    universe = build_liquidation_universe(store, fetch_liquidation_symbols())
    _write_json_atomic(raw_root / UNIVERSE_MANIFEST, universe)
    records = universe["eligible"]
    if not records:
        raise ValueError("no local/provider overlap to download")
    first_date = min(_utc(row["start"]) for row in records).date()
    last_date = max(_utc(row["end"]) for row in records).date()
    if start_date is not None:
        first_date = max(first_date, _utc(start_date).date())
    if end_date is not None:
        last_date = min(last_date, _utc(end_date).date())
    if first_date > last_date:
        raise ValueError("requested liquidation date slice has no local overlap")
    provider = CryptoHFTS3Provider(api_key, max_pool_connections=workers)
    windows = _record_windows(records)

    slice_id = f"{first_date.isoformat()}_{last_date.isoformat()}"
    manifest_path = raw_root / f"_download_manifest_{slice_id}.jsonl"
    stats = {
        "dates_indexed": 0,
        "index_dates_reused": 0,
        "objects_planned": 0,
        "objects_downloaded": 0,
        "objects_existing": 0,
        "bytes_downloaded": 0,
        "failures": [],
    }
    dates: List[str] = []
    current = first_date
    while current <= last_date:
        date = current.isoformat()
        current += timedelta(days=1)
        if _active_symbols_from_windows(windows, date):
            dates.append(date)

    # Listing one hour walks several pages because filenames are ordered by
    # symbol, not data type.  Index multiple dates in one shared pool so the
    # page walks overlap instead of serializing an entire year by day.
    index_batch_days = max(1, min(14, workers // 8))
    for offset in range(0, len(dates), index_batch_days):
        batch = dates[offset : offset + index_batch_days]
        pending = [
            date
            for date in batch
            if not (raw_root / INDEX_DIRECTORY / f"{date}.json.gz").is_file()
        ]
        stats["index_dates_reused"] += len(batch) - len(pending)
        if not pending:
            continue
        by_date: Dict[str, List[Dict[str, Any]]] = {date: [] for date in pending}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _list_hour,
                    provider,
                    date,
                    hour,
                    _active_symbols_from_windows(windows, date, hour),
                ): date
                for date in pending
                for hour in range(24)
            }
            for future in as_completed(futures):
                by_date[futures[future]].append(future.result())
        for date in pending:
            day_index = {
                "provider": PROVIDER,
                "exchange": EXCHANGE,
                "date": date,
                "listed_at": datetime.now(timezone.utc).isoformat(),
                "active_symbols": sorted(_active_symbols_from_windows(windows, date)),
                "hours": sorted(by_date[date], key=lambda row: row["hour"]),
            }
            _write_json_gzip_atomic(
                raw_root / INDEX_DIRECTORY / f"{date}.json.gz", day_index
            )
            stats["dates_indexed"] += 1

    archive_batch_days = max(1, min(4, workers // 64))
    day_workers = max(8, workers // archive_batch_days)
    for offset in range(0, len(dates), archive_batch_days):
        batch = dates[offset : offset + archive_batch_days]
        objects_by_date: Dict[str, List[Dict[str, Any]]] = {}
        for date in batch:
            day_index = _read_json_gzip(
                raw_root / INDEX_DIRECTORY / f"{date}.json.gz"
            )
            _write_daily_coverage(raw_root, day_index, windows)
            objects = [
                item
                for hour in day_index["hours"]
                for item in hour.get("liquidations", [])
            ]
            stats["objects_planned"] += len(objects)
            if objects:
                objects_by_date[date] = objects
        if not objects_by_date:
            continue

        entries: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(objects_by_date)) as executor:
            futures = {
                executor.submit(
                    _archive_day,
                    provider,
                    raw_root,
                    date,
                    objects,
                    day_workers,
                ): date
                for date, objects in objects_by_date.items()
            }
            for future in as_completed(futures):
                date = futures[future]
                try:
                    entries[date] = future.result()
                except Exception as exc:
                    stats["failures"].append(
                        {"date": date, "error": f"{type(exc).__name__}: {exc}"}
                    )
        for date in sorted(entries):
            entry = entries[date]
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(entry, ensure_ascii=False, default=_json_default)
                    + "\n"
                )
            if entry["status"] == "downloaded":
                stats["objects_downloaded"] += int(entry["member_count"])
                stats["bytes_downloaded"] += int(entry["member_bytes"])
            else:
                stats["objects_existing"] += int(entry["member_count"])
        if stats["failures"]:
            break

    result = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "universe_manifest": str(raw_root / UNIVERSE_MANIFEST),
        "download_manifest": str(manifest_path),
        "eligible_symbols": universe["eligible_count"],
        "requested_start_date": first_date.isoformat(),
        "requested_end_date": last_date.isoformat(),
        **stats,
    }
    _write_json_atomic(raw_root / f"_download_summary_{slice_id}.json", result)
    _write_json_atomic(raw_root / "_last_download_summary.json", result)
    return result


def liquidation_ingestion_status(raw_root: Path) -> Dict[str, Any]:
    """Return compact resumable-ingestion progress without touching the network."""
    root = Path(raw_root).resolve() / PROVIDER
    universe_path = root / UNIVERSE_MANIFEST
    universe = (
        json.loads(universe_path.read_text(encoding="utf-8"))
        if universe_path.is_file()
        else None
    )
    expected_dates: Set[str] = set()
    if universe:
        for record in universe.get("eligible", []):
            current = _utc(record["start"]).date()
            end = _utc(record["end"]).date()
            while current <= end:
                expected_dates.add(current.isoformat())
                current += timedelta(days=1)

    def dated(directory: str, suffix: str) -> Set[str]:
        folder = root / directory
        return {
            path.name[: -len(suffix)]
            for path in folder.glob(f"*{suffix}")
            if path.name.endswith(suffix)
        } if folder.is_dir() else set()

    indexed = dated(INDEX_DIRECTORY, ".json.gz")
    archived = dated("_daily_archives", ".tar")
    normalized = dated("_normalized_hourly", ".parquet")
    covered = dated("_coverage_hourly", ".parquet")
    archives = list((root / "_daily_archives").glob("*.tar"))
    return {
        "raw_root": str(root),
        "eligible_symbols": universe.get("eligible_count") if universe else None,
        "expected_dates": len(expected_dates),
        "indexed_dates": len(indexed),
        "archived_dates": len(archived),
        "normalized_dates": len(normalized),
        "coverage_dates": len(covered),
        "archive_bytes": sum(path.stat().st_size for path in archives),
        "next_index_date": min(expected_dates - indexed) if expected_dates - indexed else None,
        "next_archive_date": min(expected_dates - archived) if expected_dates - archived else None,
        "ready_for_alignment": bool(expected_dates)
        and expected_dates <= indexed
        and expected_dates <= covered
        and all(
            not any(
                hour.get("liquidations")
                for hour in _read_json_gzip(
                    root / INDEX_DIRECTORY / f"{date}.json.gz"
                ).get("hours", [])
            )
            or date in archived and date in normalized
            for date in expected_dates
        ),
    }


def audit_liquidation_history(raw_root: Path) -> Dict[str, Any]:
    """Check daily raw archive and normalized aggregate metadata."""
    root = Path(raw_root).resolve() / PROVIDER
    status = liquidation_ingestion_status(raw_root)
    manifests = sorted((root / "_daily_manifests").glob("*.json"))
    issues: List[Dict[str, Any]] = []
    member_count = 0
    member_bytes = 0
    invalid_notional_count = 0
    invalid_objects: List[Dict[str, Any]] = []
    aggregate_rows = 0
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        date = str(payload.get("date") or manifest_path.stem)
        archive = Path(str(payload.get("archive", "")))
        aggregate = Path(str(payload.get("hourly_aggregate", "")))
        if not archive.is_file():
            issues.append({"date": date, "issue": "archive_missing"})
            continue
        if archive.stat().st_size != int(payload.get("archive_bytes", -1)):
            issues.append({"date": date, "issue": "archive_size_mismatch"})
        if not aggregate.is_file():
            issues.append({"date": date, "issue": "hourly_aggregate_missing"})
        elif aggregate.stat().st_size != int(
            payload.get("hourly_aggregate_bytes", -1)
        ):
            issues.append({"date": date, "issue": "hourly_aggregate_size_mismatch"})
        members = payload.get("members", [])
        member_count += int(payload.get("member_count", len(members)))
        member_bytes += int(payload.get("member_bytes", 0))
        aggregate_rows += int(payload.get("hourly_aggregate_rows", 0))
        for member in members:
            count = int(member.get("invalid_notional_count", 0))
            if count:
                invalid_notional_count += count
                invalid_objects.append(
                    {
                        "date": date,
                        "key": member.get("key"),
                        "invalid_notional_count": count,
                    }
                )

    result = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "provider": PROVIDER,
        "exchange": EXCHANGE,
        "status": "ready" if status["ready_for_alignment"] and not issues else "failed",
        "eligible_symbols": status["eligible_symbols"],
        "expected_dates": status["expected_dates"],
        "verified_daily_manifests": len(manifests),
        "raw_archive_bytes": status["archive_bytes"],
        "raw_member_count": member_count,
        "raw_member_bytes": member_bytes,
        "hourly_aggregate_rows": aggregate_rows,
        "invalid_notional_count": invalid_notional_count,
        "invalid_objects": invalid_objects,
        "issues": issues,
    }
    _write_json_atomic(root / "_audit_summary.json", result)
    return result


def _provider_parquet_table(payload: bytes):
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "liquidation alignment requires the 'liquidation-data' extra"
        ) from exc
    if payload.startswith(b"\x28\xb5\x2f\xfd"):
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "outer-zstd liquidation files require the 'liquidation-data' extra"
            ) from exc
        payload = zstandard.ZstdDecompressor().decompress(payload)
    return parquet.read_table(io.BytesIO(payload))


def _read_provider_parquet_bytes(payload: bytes) -> pd.DataFrame:
    return _provider_parquet_table(payload).to_pandas()


def _numeric_array(values: Sequence[object]) -> np.ndarray:
    output = np.empty(len(values), dtype=float)
    for index, value in enumerate(values):
        try:
            output[index] = float(value)
        except (TypeError, ValueError):
            output[index] = np.nan
    return output


def _aggregate_liquidation_payload(
    payload: bytes,
    *,
    symbol: str,
    source_key: str,
) -> tuple[List[Dict[str, Any]], int]:
    """Aggregate one small provider object without per-file pandas groupby cost."""
    table = _provider_parquet_table(payload)
    required = {
        "event_time", "received_time", "side", "price", "average_price",
        "quantity", "filled_quantity",
    }
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError("liquidation data missing columns: " + ", ".join(missing))
    if table.num_rows == 0:
        return [], 0
    event_time = np.asarray(table["event_time"].combine_chunks()).astype(np.int64)
    received_time = np.asarray(table["received_time"].combine_chunks()).astype(np.int64)
    sides = np.asarray(
        [str(value).upper() for value in table["side"].to_pylist()], dtype=object
    )
    if not np.isin(sides, ["BUY", "SELL"]).all():
        raise ValueError("liquidation data contains unsupported side values")
    price = _numeric_array(table["price"].to_pylist())
    average_price = _numeric_array(table["average_price"].to_pylist())
    quantity = _numeric_array(table["quantity"].to_pylist())
    filled_quantity = _numeric_array(table["filled_quantity"].to_pylist())
    effective_price = np.where(average_price > 0, average_price, price)
    effective_quantity = np.where(filled_quantity > 0, filled_quantity, quantity)
    notional = effective_price * effective_quantity
    invalid_notional = ~np.isfinite(notional) | (notional < 0)
    invalid_notional_count = int(invalid_notional.sum())
    notional[invalid_notional] = np.nan
    hour_ids = event_time // 3_600_000
    rows: List[Dict[str, Any]] = []
    for hour_id in np.unique(hour_ids):
        selected = hour_ids == hour_id
        sell = selected & (sides == "SELL")
        buy = selected & (sides == "BUY")

        def notional_sum(mask: np.ndarray) -> float:
            values = notional[mask]
            if not mask.any():
                return 0.0
            return float(np.nansum(values)) if np.isfinite(values).any() else np.nan

        rows.append(
            {
                "symbol": symbol,
                "timestamp": pd.to_datetime(
                    int(hour_id) * 3_600_000, unit="ms", utc=True
                ),
                "liquidation_event_count": int(selected.sum()),
                "long_liquidation_count": int(sell.sum()),
                "short_liquidation_count": int(buy.sum()),
                "long_liquidation_notional_usdt": notional_sum(sell),
                "short_liquidation_notional_usdt": notional_sum(buy),
                "liquidation_event_at_min": pd.to_datetime(
                    int(event_time[selected].min()), unit="ms", utc=True
                ),
                "liquidation_event_at_max": pd.to_datetime(
                    int(event_time[selected].max()), unit="ms", utc=True
                ),
                "liquidation_received_at_max": pd.to_datetime(
                    int(received_time[selected].max()), unit="ns", utc=True
                ),
                "source_key": source_key,
            }
        )
    return rows, invalid_notional_count


def aggregate_liquidation_events(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized event rows to causal UTC hours."""
    required = {
        "event_time",
        "received_time",
        "side",
        "price",
        "average_price",
        "quantity",
        "filled_quantity",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError("liquidation data missing columns: " + ", ".join(missing))
    if events.empty:
        return pd.DataFrame()
    frame = events.copy()
    frame["event_at"] = pd.to_datetime(frame["event_time"], unit="ms", utc=True)
    frame["received_at"] = pd.to_datetime(
        frame["received_time"], unit="ns", utc=True
    )
    frame["side"] = frame["side"].astype(str).str.upper()
    invalid_side = ~frame["side"].isin(["BUY", "SELL"])
    if invalid_side.any():
        raise ValueError("liquidation data contains unsupported side values")
    for column in ("price", "average_price", "quantity", "filled_quantity"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    price = frame["average_price"].where(frame["average_price"] > 0, frame["price"])
    quantity = frame["filled_quantity"].where(
        frame["filled_quantity"] > 0, frame["quantity"]
    )
    frame["notional_usdt"] = price * quantity
    if (~np.isfinite(frame["notional_usdt"]) | (frame["notional_usdt"] < 0)).any():
        raise ValueError("liquidation data contains invalid notional values")
    frame["hour"] = frame["event_at"].dt.floor("h")
    frame["long_notional_usdt"] = frame["notional_usdt"].where(
        frame["side"] == "SELL", 0.0
    )
    frame["short_notional_usdt"] = frame["notional_usdt"].where(
        frame["side"] == "BUY", 0.0
    )
    frame["long_count"] = (frame["side"] == "SELL").astype(int)
    frame["short_count"] = (frame["side"] == "BUY").astype(int)
    grouped = frame.groupby("hour", sort=True).agg(
        liquidation_event_count=("side", "size"),
        long_liquidation_count=("long_count", "sum"),
        short_liquidation_count=("short_count", "sum"),
        long_liquidation_notional_usdt=("long_notional_usdt", "sum"),
        short_liquidation_notional_usdt=("short_notional_usdt", "sum"),
        liquidation_event_at_min=("event_at", "min"),
        liquidation_event_at_max=("event_at", "max"),
        liquidation_received_at_max=("received_at", "max"),
    )
    grouped.index.name = "timestamp"
    return grouped


def align_liquidation_history(
    db_path: Path,
    raw_root: Path,
    output_root: Path,
) -> Dict[str, Any]:
    """Causally join raw liquidations to local OHLCV, funding, and OI."""
    raw_root = Path(raw_root).resolve() / PROVIDER
    universe_path = raw_root / UNIVERSE_MANIFEST
    if not universe_path.is_file():
        raise FileNotFoundError(f"missing liquidation universe: {universe_path}")
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    store = MarketDataStore(Path(db_path))
    engine = FactorEngine(store)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    try:
        import pyarrow.dataset as arrow_dataset
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "liquidation alignment requires the 'liquidation-data' extra"
        ) from exc
    coverage_files = sorted((raw_root / "_coverage_hourly").glob("*.parquet"))
    event_files = sorted((raw_root / "_normalized_hourly").glob("*.parquet"))
    if not coverage_files:
        raise FileNotFoundError("no hourly liquidation coverage files are present")
    coverage_dataset = arrow_dataset.dataset(
        [str(path) for path in coverage_files], format="parquet"
    )
    event_dataset = (
        arrow_dataset.dataset([str(path) for path in event_files], format="parquet")
        if event_files
        else None
    )
    raw_audit_path = raw_root / "_audit_summary.json"
    raw_audit = (
        json.loads(raw_audit_path.read_text(encoding="utf-8"))
        if raw_audit_path.is_file()
        else {}
    )
    invalid_notional_by_key = {
        str(row["key"]): int(row["invalid_notional_count"])
        for row in raw_audit.get("invalid_objects", [])
    }

    for record in universe["eligible"]:
        symbol = record["symbol"]
        coverage = coverage_dataset.to_table(
            filter=arrow_dataset.field("symbol") == symbol
        ).to_pandas()
        if coverage.empty:
            results.append({"symbol": symbol, "status": "missing_hourly_coverage"})
            continue
        coverage["timestamp"] = pd.to_datetime(coverage["timestamp"], utc=True)
        coverage = coverage.loc[
            (coverage["timestamp"] >= _utc(record["start"]))
            & (coverage["timestamp"] <= _utc(record["end"]))
        ].drop(columns=["symbol"]).set_index("timestamp").sort_index()

        if event_dataset is not None:
            event_rows = event_dataset.to_table(
                filter=arrow_dataset.field("symbol") == symbol
            ).to_pandas()
        else:
            event_rows = pd.DataFrame()
        if event_rows.empty:
            aggregated = pd.DataFrame(
                index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
                columns=[
                    "liquidation_event_count",
                    "long_liquidation_count",
                    "short_liquidation_count",
                    "long_liquidation_notional_usdt",
                    "short_liquidation_notional_usdt",
                    "liquidation_event_at_min",
                    "liquidation_event_at_max",
                    "liquidation_received_at_max",
                    "normalized_source_file_count",
                    "partition_mismatch_source_file_count",
                    "invalid_notional_count",
                ],
            )
        else:
            event_rows["timestamp"] = pd.to_datetime(
                event_rows["timestamp"], utc=True
            )
            partition_parts = event_rows["source_key"].str.split("/")
            event_rows["source_partition_at"] = pd.to_datetime(
                partition_parts.str[1]
                + "T"
                + partition_parts.str[2]
                + ":00:00Z",
                utc=True,
            )
            event_rows["partition_mismatch"] = (
                event_rows["source_partition_at"] != event_rows["timestamp"]
            ).astype(int)
            event_rows = event_rows.sort_values(["source_key", "timestamp"])
            event_rows["invalid_notional_count"] = event_rows["source_key"].map(
                invalid_notional_by_key
            ).fillna(0).astype(int)
            event_rows.loc[
                event_rows.duplicated("source_key", keep="first"),
                "invalid_notional_count",
            ] = 0
            aggregated = event_rows.groupby("timestamp", sort=True).agg(
                liquidation_event_count=("liquidation_event_count", "sum"),
                long_liquidation_count=("long_liquidation_count", "sum"),
                short_liquidation_count=("short_liquidation_count", "sum"),
                long_liquidation_notional_usdt=(
                    "long_liquidation_notional_usdt",
                    lambda values: values.sum(min_count=1),
                ),
                short_liquidation_notional_usdt=(
                    "short_liquidation_notional_usdt",
                    lambda values: values.sum(min_count=1),
                ),
                liquidation_event_at_min=("liquidation_event_at_min", "min"),
                liquidation_event_at_max=("liquidation_event_at_max", "max"),
                liquidation_received_at_max=(
                    "liquidation_received_at_max", "max"
                ),
                normalized_source_file_count=("source_key", "nunique"),
                partition_mismatch_source_file_count=(
                    "partition_mismatch", "sum"
                ),
                invalid_notional_count=("invalid_notional_count", "sum"),
            )

        factors = engine.load(
            symbol,
            interval="1h",
            start=record["start"],
            end=record["end"],
            base_market=USD_M_PERPETUAL,
            warmup_days=30,
        )
        factor_columns = [
            "open", "high", "low", "close", "volume", "quote_volume",
            "trades", "close_time", "available_at", "funding_observed_at",
            "funding_rate", "funding_interval_hours", "metrics_observed_at",
            "open_interest", "open_interest_value",
        ]
        aligned = factors[[column for column in factor_columns if column in factors]].copy()
        aligned = aligned.rename(
            columns={
                "open": "futures_open", "high": "futures_high",
                "low": "futures_low", "close": "futures_close",
                "volume": "futures_volume", "quote_volume": "futures_quote_volume",
                "trades": "futures_trades", "close_time": "futures_close_time",
            }
        )
        spot = store.load_bars(
            SPOT,
            resolve_market_symbols(symbol).spot,
            interval="1h",
            start=record["start"],
            end=record["end"],
        )[["open", "high", "low", "close", "volume", "quote_volume", "trades", "close_time"]]
        spot = spot.add_prefix("spot_")
        aligned = aligned.join(spot, how="inner").join(coverage, how="left")
        aligned = aligned.join(aggregated, how="left")

        missing_event_payload = aligned["liquidation_coverage_status"].eq(
            "valid_event"
        ) & aligned["liquidation_event_count"].isna()
        has_companion = aligned["liquidation_companion_types"].fillna("").ne("")
        aligned.loc[
            missing_event_payload & has_companion,
            "liquidation_coverage_status",
        ] = "valid_zero"
        aligned.loc[
            missing_event_payload & ~has_companion,
            "liquidation_coverage_status",
        ] = "source_unknown"
        event_present = aligned["liquidation_event_count"].notna()
        aligned.loc[event_present, "liquidation_coverage_status"] = "valid_event"
        aligned["liquidation_partition_mismatch"] = (
            missing_event_payload
            | aligned["partition_mismatch_source_file_count"].fillna(0).gt(0)
        )

        event_columns = [
            "liquidation_event_count", "long_liquidation_count",
            "short_liquidation_count", "long_liquidation_notional_usdt",
            "short_liquidation_notional_usdt", "invalid_notional_count",
        ]
        valid_zero = aligned["liquidation_coverage_status"].eq("valid_zero")
        for column in event_columns:
            aligned.loc[valid_zero, column] = 0.0
        aligned["liquidation_available_at"] = aligned[
            ["available_at", "liquidation_received_at_max"]
        ].max(axis=1)
        aligned.insert(0, "symbol", symbol)
        aligned.insert(1, "market", EXCHANGE)

        destination = output_root / f"symbol={symbol}" / "aligned_1h.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        aligned.to_parquet(temporary, index=True, compression="zstd")
        os.replace(temporary, destination)
        results.append(
            {
                "symbol": symbol,
                "status": "aligned",
                "rows": len(aligned),
                "valid_event_hours": int(
                    aligned["liquidation_coverage_status"].eq("valid_event").sum()
                ),
                "valid_zero_hours": int(valid_zero.sum()),
                "source_unknown_hours": int(
                    aligned["liquidation_coverage_status"].eq("source_unknown").sum()
                ),
                "partition_mismatch_hours": int(
                    aligned["liquidation_partition_mismatch"].sum()
                ),
                "invalid_notional_hours": int(
                    aligned["invalid_notional_count"].fillna(0).gt(0).sum()
                ),
                "start": _iso(aligned.index.min()),
                "end": _iso(aligned.index.max()),
                "path": str(destination),
            }
        )

    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "provider": PROVIDER,
        "exchange": EXCHANGE,
        "input_universe": str(universe_path),
        "output_root": str(output_root),
        "symbols_total": len(results),
        "symbols_aligned": sum(row["status"] == "aligned" for row in results),
        "results": results,
    }
    _write_json_atomic(output_root / "alignment_summary.json", summary)
    return summary


def audit_liquidation_alignment(output_root: Path) -> Dict[str, Any]:
    """Audit causal and structural invariants across every aligned symbol file."""
    output_root = Path(output_root).resolve()
    summary_path = output_root / "alignment_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing alignment summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    files = sorted(output_root.glob("symbol=*/aligned_1h.parquet"))
    issues: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    mismatch_timestamps: Counter[str] = Counter()
    unknown_timestamps: Counter[str] = Counter()
    totals = Counter()

    for path in files:
        frame = pd.read_parquet(path)
        symbol = path.parent.name.removeprefix("symbol=")
        totals["rows"] += len(frame)
        totals["bytes"] += path.stat().st_size
        if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            issues.append({"symbol": symbol, "issue": "timestamp_index_invalid"})
        if frame["symbol"].astype(str).nunique() != 1 or str(
            frame["symbol"].iloc[0]
        ) != symbol:
            issues.append({"symbol": symbol, "issue": "symbol_column_mismatch"})

        for observed_at in ("funding_observed_at", "metrics_observed_at"):
            future = frame[observed_at].notna() & (
                frame[observed_at] > frame["available_at"]
            )
            if future.any():
                issues.append(
                    {
                        "symbol": symbol,
                        "issue": f"future_{observed_at}",
                        "rows": int(future.sum()),
                    }
                )
        invalid_available = frame["liquidation_received_at_max"].notna() & (
            frame["liquidation_received_at_max"]
            > frame["liquidation_available_at"]
        )
        if invalid_available.any():
            issues.append(
                {
                    "symbol": symbol,
                    "issue": "liquidation_available_at_before_receive",
                    "rows": int(invalid_available.sum()),
                }
            )
        negative_notional = (
            frame[
                [
                    "long_liquidation_notional_usdt",
                    "short_liquidation_notional_usdt",
                ]
            ]
            < 0
        ).any(axis=1)
        if negative_notional.any():
            issues.append(
                {
                    "symbol": symbol,
                    "issue": "negative_aligned_notional",
                    "rows": int(negative_notional.sum()),
                }
            )

        valid_zero = frame["liquidation_coverage_status"].eq("valid_zero")
        zero_columns = [
            "liquidation_event_count",
            "long_liquidation_count",
            "short_liquidation_count",
            "long_liquidation_notional_usdt",
            "short_liquidation_notional_usdt",
        ]
        nonzero_zero_hours = valid_zero & frame[zero_columns].fillna(0).ne(0).any(axis=1)
        if nonzero_zero_hours.any():
            issues.append(
                {
                    "symbol": symbol,
                    "issue": "valid_zero_contains_event_value",
                    "rows": int(nonzero_zero_hours.sum()),
                }
            )

        counts = frame["liquidation_coverage_status"].value_counts(dropna=False)
        for status_name, count in counts.items():
            status_counts[str(status_name)] += int(count)
        totals["funding_missing_rows"] += int(frame["funding_rate"].isna().sum())
        totals["oi_missing_rows"] += int(frame["open_interest"].isna().sum())
        totals["partition_mismatch_hours"] += int(
            frame["liquidation_partition_mismatch"].sum()
        )
        event_rows = frame["liquidation_event_count"].fillna(0).gt(0)
        totals["event_hours"] += int(event_rows.sum())
        totals["liquidation_events"] += int(
            frame["liquidation_event_count"].fillna(0).sum()
        )
        totals["event_hours_with_missing_notional"] += int(
            (
                event_rows
                & frame[
                    [
                        "long_liquidation_notional_usdt",
                        "short_liquidation_notional_usdt",
                    ]
                ].isna().any(axis=1)
            ).sum()
        )
        totals["invalid_notional_count"] += int(
            frame["invalid_notional_count"].fillna(0).sum()
        )
        totals["invalid_notional_hours"] += int(
            frame["invalid_notional_count"].fillna(0).gt(0).sum()
        )
        for timestamp in frame.index[frame["liquidation_partition_mismatch"]]:
            mismatch_timestamps[_iso(timestamp)] += 1
        for timestamp in frame.index[
            frame["liquidation_coverage_status"].eq("source_unknown")
        ]:
            unknown_timestamps[_iso(timestamp)] += 1

    expected_symbols = int(summary.get("symbols_aligned", 0))
    if len(files) != expected_symbols:
        issues.append(
            {
                "issue": "aligned_file_count_mismatch",
                "expected": expected_symbols,
                "actual": len(files),
            }
        )
    result = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if not issues else "failed",
        "aligned_files": len(files),
        "aligned_rows": totals["rows"],
        "aligned_bytes": totals["bytes"],
        "coverage_status_counts": dict(sorted(status_counts.items())),
        "event_hours": totals["event_hours"],
        "liquidation_events": totals["liquidation_events"],
        "funding_missing_rows": totals["funding_missing_rows"],
        "oi_missing_rows": totals["oi_missing_rows"],
        "partition_mismatch_hours": totals["partition_mismatch_hours"],
        "event_hours_with_missing_notional": totals[
            "event_hours_with_missing_notional"
        ],
        "invalid_notional_count": totals["invalid_notional_count"],
        "invalid_notional_hours": totals["invalid_notional_hours"],
        "top_partition_mismatch_timestamps": mismatch_timestamps.most_common(20),
        "top_source_unknown_timestamps": unknown_timestamps.most_common(20),
        "issues": issues,
    }
    _write_json_atomic(output_root / "alignment_audit.json", result)
    return result
