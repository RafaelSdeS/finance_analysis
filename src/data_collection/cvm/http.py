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
# CVM open-data zip floor -- earlier years 404 and are skipped. Does NOT mean
# every field is usable from 2010: FCA's own Codigo_Negociacao (trading code)
# field is verified 100% blank for every filer, every year, 2010-2017 (live
# re-check 2026-08-15 against the real CVM zips, not just this repo's cache)
# -- crosswalk.py's ticker recovery has a real floor of 2018, four years past
# this constant.
START_YEAR = 2010
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


def fetch_csv_url(url: str) -> str | None:
    """Retry-GET for a single non-zipped CVM CSV (e.g. cad_cia_aberta.csv, the
    static company-master registry) -- same retry/timeout policy as fetch_zip,
    for the one CVM source that isn't a yearly zip. Returns decoded text, or
    None on a 404/exhausted-retries."""
    for attempt in range(RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            break
        except requests.RequestException as e:
            if attempt == RETRIES:
                log.warning("fetch_csv_url %s: network error after %d attempts: %s",
                            url, RETRIES + 1, e)
                return None
            log.warning("fetch_csv_url %s: %s — retrying (%d/%d)",
                        url, type(e).__name__, attempt + 1, RETRIES)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.content.decode("latin-1")


def read_csv(zf: zipfile.ZipFile, name: str) -> list[dict]:
    try:
        raw = zf.read(name).decode("latin-1")
    except KeyError:
        return []
    return list(csv.DictReader(io.StringIO(raw), delimiter=";"))


def digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")
