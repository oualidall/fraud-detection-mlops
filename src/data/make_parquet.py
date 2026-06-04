"""Convert the raw IEEE-CIS CSV files to compact Parquet files.

Why this exists
---------------
The raw CSVs are large (``train_transaction.csv`` alone is ~650 MB) and loading
them naively with ``pandas.read_csv`` materialises everything as int64/float64,
which can spike to several GB of RAM and crash low-memory machines (e.g. WSL2
capped at half the host RAM).

This script reads each CSV **in streaming blocks** with PyArrow and writes a
downcast (int32/float32) Parquet file. Memory stays bounded (~a few hundred MB)
regardless of file size, and the resulting Parquet files are several times
smaller and load in seconds.

Usage::

    python -m src.data.make_parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)

# 64 MB read blocks keep peak memory low while streaming.
BLOCK_SIZE = 64 * 1024 * 1024

DATASETS = ("train_transaction", "train_identity", "test_transaction", "test_identity")


def _downcast_batch(batch: pa.RecordBatch) -> pa.Table:
    """Cast int64 -> int32 and float64 -> float32 to shrink the table."""
    columns = []
    names = []
    for field in batch.schema:
        column = batch.column(field.name)
        if pa.types.is_int64(field.type):
            column = column.cast(pa.int32(), safe=False)
        elif pa.types.is_float64(field.type):
            column = column.cast(pa.float32(), safe=False)
        columns.append(column)
        names.append(field.name)
    return pa.table(columns, names=names)


def convert(name: str, *, overwrite: bool = False) -> Path | None:
    """Stream-convert one CSV to Parquet. Returns the output path (or None)."""
    source = RAW_DATA_DIR / f"{name}.csv"
    if not source.exists():
        logger.warning("skipping %s: %s not found", name, source)
        return None

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = PROCESSED_DATA_DIR / f"{name}.parquet"
    if destination.exists() and not overwrite:
        logger.info("skipping %s: %s already exists", name, destination)
        return destination

    logger.info("converting %s -> %s (streaming)", source.name, destination.name)
    reader = pacsv.open_csv(
        source,
        read_options=pacsv.ReadOptions(block_size=BLOCK_SIZE),
    )

    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for batch in reader:
            table = _downcast_batch(batch)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="snappy")
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
        reader.close()

    size_mb = destination.stat().st_size / 1e6
    logger.info("wrote %s rows to %s (%.1f MB)", rows, destination.name, size_mb)
    return destination


def main() -> None:
    for name in DATASETS:
        convert(name)
    logger.info("parquet conversion complete")


if __name__ == "__main__":
    main()
