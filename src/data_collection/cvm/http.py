"""cvm/http.py — shared CVM open-data download plumbing.

Every CVM open-data source (FCA crosswalk, DFP/ITR statements, FRE shares,
the ITR/DFP filing-date registers) is a yearly zip at the same URL shape,
containing one or more semicolon-delimited, latin-1 CSVs. This is the one
download-with-retry implementation; each step module reads whichever CSV
member(s) it needs out of the zip via read_csv().
"""

import csv
import io
import logging
import re
import zipfile

import requests

log = logging.getLogger("cvm")

DOC_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{doc}/DADOS/{doc_l}_cia_aberta_{year}.zip"
START_YEAR = 2010  # CVM open-data coverage floor; earlier years 404 and are skipped
TIMEOUT = (15, 120)  # (connect, read) — fail fast on a stalled CVM connection
RETRIES = 2


def fetch_zip(doc: str, year: int) -> zipfile.ZipFile | None:
    """One CVM yearly zip (FCA/DFP/ITR/FRE); None when the year isn't published (404)."""
    url = DOC_URL.format(doc=doc.upper(), doc_l=doc.lower(), year=year)
    log.info("%s %d: downloading...", doc, year)
    for attempt in range(RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            break
        except requests.RequestException as e:
            if attempt == RETRIES:
                log.warning("%s %d: network error after %d attempts: %s",
                            doc, year, RETRIES + 1, e)
                return None
            log.warning("%s %d: %s — retrying (%d/%d)", doc, year,
                        type(e).__name__, attempt + 1, RETRIES)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    try:
        return zipfile.ZipFile(io.BytesIO(resp.content))
    except zipfile.BadZipFile as e:
        log.warning("%s %d: corrupt zip: %s", doc, year, e)
        return None


def read_csv(zf: zipfile.ZipFile, name: str) -> list[dict]:
    try:
        raw = zf.read(name).decode("latin-1")
    except KeyError:
        return []
    return list(csv.DictReader(io.StringIO(raw), delimiter=";"))


def digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")
