from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 500


class DataValidationError(ValueError):
    """An upload cannot be safely analyzed by the prototype."""


def _read_bytes(source: bytes | bytearray | BinaryIO | str | Path) -> bytes:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataValidationError("The selected file does not exist.")
        if path.stat().st_size > MAX_UPLOAD_BYTES:
            raise DataValidationError("CSV exceeds the 20 MB V1 upload limit.")
        return path.read_bytes()
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    value = source.read()
    return value.encode() if isinstance(value, str) else value


def load_csv(source: bytes | bytearray | BinaryIO | str | Path) -> pd.DataFrame:
    raw = _read_bytes(source)
    if not raw or not raw.strip():
        raise DataValidationError("The CSV is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise DataValidationError("CSV exceeds the 20 MB V1 upload limit.")
    try:
        frame = pd.read_csv(BytesIO(raw), nrows=MAX_ROWS + 1)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataValidationError(f"Could not parse a valid UTF-8 CSV: {exc}") from exc
    if frame.empty or len(frame.columns) == 0:
        raise DataValidationError("The CSV contains no data rows.")
    if len(frame) > MAX_ROWS:
        raise DataValidationError(f"CSV exceeds the {MAX_ROWS:,}-row V1 analysis limit.")
    if len(frame.columns) > MAX_COLUMNS:
        raise DataValidationError(f"CSV exceeds the {MAX_COLUMNS}-column V1 analysis limit.")
    if any(str(c).strip() == "" for c in frame.columns):
        raise DataValidationError("Every column must have a name.")
    return frame
