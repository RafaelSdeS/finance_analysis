"""
compress_archive_cache.py — one-time migration of sec/http.py's on-disk
Archive-document cache (data/raw/us/sec/archive_cache/) from plain .txt to
gzip-compressed .txt.gz.

Background (2026-08-15): the cache grew to 35GB / 62k+ files of plain-text
SEC filings (highly compressible SGML/HTML). sec/http.py now writes and reads
.txt.gz directly (gzip.open), so newly cached documents are already
compressed -- this script converts the existing .txt entries so the cache
stays warm instead of every pre-existing entry becoming a permanent miss
(silently re-downloaded once, then cached again under the new suffix).

Idempotent/resumable: skips any .txt whose .txt.gz twin already exists, so
it's safe to re-run if interrupted partway through.

Run from project root: python -m src.data_collection.one_off.compress_archive_cache
"""

import gzip
import logging

from ..sec.http import ARCHIVE_CACHE_DIR

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    files = sorted(ARCHIVE_CACHE_DIR.glob("*.txt"))
    log.info("found %d .txt files to convert in %s", len(files), ARCHIVE_CACHE_DIR)

    done = skipped = 0
    before_total = after_total = 0
    for i, path in enumerate(files, 1):
        gz_path = path.with_suffix(".txt.gz")
        if gz_path.exists():
            skipped += 1
            continue

        original = path.read_bytes()
        with gzip.open(gz_path, "wb") as f:
            f.write(original)

        with gzip.open(gz_path, "rb") as f:
            if f.read() != original:
                raise ValueError(f"{path}: gzip round-trip mismatch, aborting -- {gz_path} left for inspection")

        before_total += len(original)
        after_total += gz_path.stat().st_size
        path.unlink()
        done += 1

        if i % 2000 == 0 or i == len(files):
            ratio = (1 - after_total / before_total) * 100 if before_total else 0
            log.info("%d/%d done (%d skipped) -- %.1fGB -> %.1fGB so far (%.0f%% smaller)",
                      i, len(files), skipped, before_total / 1e9, after_total / 1e9, ratio)

    log.info("finished: %d converted, %d already done -- %.1fGB -> %.1fGB",
              done, skipped, before_total / 1e9, after_total / 1e9)


if __name__ == "__main__":
    main()
