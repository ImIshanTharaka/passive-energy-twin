#!/usr/bin/env python3
"""
download_epw.py - Resolve and download the EPW weather file set (Sub-task 2.3).

Reads config/climate_config.yaml, resolves each station's download URL from
climate.onebuilding.org by WMO number, selects a single data vintage common to
every station, downloads and extracts the files, validates them, and writes a
provenance record.

WHY RESOLVE INSTEAD OF HARDCODE
-------------------------------
Station filenames upstream embed the station name verbatim
('USA_IL_Chicago.OHare.Intl.AP.725300_TMYx.2011-2025.zip'). Those spellings
change between releases. The WMO number does not. This script matches
'.{wmo}_TMYx' against the live directory listing, so a rename upstream cannot
silently break the campaign.

WHY A COMMON VINTAGE
--------------------
Mixing TMY3 and TMYx files across the set is what confounded the earlier
held-out London test: London differed from the training climates in data
vintage as well as in climate. This script refuses to mix vintages. It probes
every station first, then picks the newest vintage that ALL stations offer.

USAGE
-----
    python climate/download_epw.py                 # probe, then download
    python climate/download_epw.py --dry-run       # probe and report only
    python climate/download_epw.py --vintage 2009-2023
    python climate/download_epw.py --only colombo --force

OUTPUTS
-------
    climate/epw_files/<canonical>.epw     one per climate
    climate/epw_stat/<canonical>.stat     ASHRAE design conditions + published
                                          degree-days, used as the independent
                                          cross-check in Sub-task 2.3.4
    climate/epw_provenance.json           URL, vintage, SHA-256, size, row
                                          count per file — the reproducibility
                                          record cited in the thesis
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    sys.exit(
        "\nPyYAML is not available to this interpreter:\n"
        f"    {sys.executable}\n\n"
        "This usually means the project venv is not active. Run it explicitly:\n"
        "    ./venv/bin/python climate/download_epw.py --dry-run\n\n"
        "If that also fails, the venv needs rebuilding — see README.\n"
    )

BASE_URL = "https://climate.onebuilding.org"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "climate_config.yaml"
EPW_DIR = REPO_ROOT / "climate" / "epw_files"
STAT_DIR = REPO_ROOT / "climate" / "epw_stat"
PROVENANCE_PATH = REPO_ROOT / "climate" / "epw_provenance.json"

EXPECTED_ROWS = 8760          # TMY files never contain 29 February
EPW_HEADER_LINES = 8
REQUEST_TIMEOUT = 90
MAX_RETRIES = 3
RETRY_BACKOFF = 4.0           # seconds, multiplied by attempt number


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

_SSL_CONTEXT: ssl.SSLContext | None = None


def _ssl_context() -> ssl.SSLContext:
    """TLS context backed by an explicit CA bundle.

    The python.org macOS framework build does not read the system keychain.
    Until 'Install Certificates.command' is run it has no root store at all,
    and every HTTPS request fails with CERTIFICATE_VERIFY_FAILED. Pinning
    certifi's bundle makes this script work regardless of machine state —
    which matters when the repository has to run on someone else's machine
    for reproducibility.
    """
    global _SSL_CONTEXT
    if _SSL_CONTEXT is None:
        try:
            import certifi
            _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
        except ModuleNotFoundError:
            _SSL_CONTEXT = ssl.create_default_context()
    return _SSL_CONTEXT


def _fetch(url: str) -> bytes:
    """GET with a browser User-Agent and bounded retries."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT, context=_ssl_context()
            ) as response:
                return response.read()
        except urllib.error.URLError as exc:
            # A certificate failure will never succeed on retry — fail fast
            # with the fix rather than burning the backoff schedule.
            if isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
                raise SystemExit(
                    f"\nTLS certificate verification failed for {url}\n\n"
                    "This interpreter has no usable CA root store. Fix either way:\n\n"
                    "  1. Install certifi into the venv (preferred — self-contained):\n"
                    "       ./venv/bin/python -m pip install certifi\n\n"
                    "  2. Or run the python.org one-time certificate installer:\n"
                    "       open '/Applications/Python 3.13/Install Certificates.command'\n"
                ) from exc
            last_error = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF * attempt
                print(f"      retry {attempt}/{MAX_RETRIES - 1} in {delay:.0f}s ({exc})")
                time.sleep(delay)
        except (urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF * attempt
                print(f"      retry {attempt}/{MAX_RETRIES - 1} in {delay:.0f}s ({exc})")
                time.sleep(delay)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def list_station_vintages(onebuilding_dir: str, wmo: str) -> dict[str, str]:
    """Return {vintage: absolute_url} for every TMYx file matching this WMO.

    The base TMYx file (no explicit period) is keyed as 'all-years'.
    """
    listing_url = f"{BASE_URL}/{onebuilding_dir.strip('/')}/"
    html = _fetch(listing_url).decode("utf-8", errors="replace")

    pattern = re.compile(
        rf'href\s*=\s*["\']([^"\']*\.{re.escape(wmo)}_TMYx(?:\.(\d{{4}}-\d{{4}}))?\.zip)["\']',
        re.IGNORECASE,
    )

    found: dict[str, str] = {}
    for href, vintage in pattern.findall(html):
        filename = href.split("/")[-1]
        found[vintage or "all-years"] = f"{listing_url}{filename}"
    return found


def list_directory(onebuilding_dir: str, grep: str | None, vintage: str | None) -> int:
    """Print every TMYx station in an upstream directory.

    Used when a configured WMO number resolves to nothing — usually because
    the station is not in the TMYx set at all. TMYx is built from ISD hourly
    records, so co-operative and observatory sites that TMY3 included may have
    no TMYx counterpart, and a nearby airport has to be substituted.
    """
    listing_url = f"{BASE_URL}/{onebuilding_dir.strip('/')}/"
    print(f"\nListing {listing_url}")
    if grep:
        print(f"Filtering on: {grep!r}")

    html = _fetch(listing_url).decode("utf-8", errors="replace")
    pattern = re.compile(
        r'href\s*=\s*["\']([^"\']*?\.(\d{6})_TMYx(?:\.(\d{4}-\d{4}))?\.zip)["\']',
        re.IGNORECASE,
    )

    rows = []
    for href, wmo, vint in pattern.findall(html):
        filename = href.split("/")[-1]
        if vintage and (vint or "all-years") != vintage:
            continue
        if grep and grep.lower() not in filename.lower():
            continue
        rows.append((wmo, vint or "all-years", filename))

    if not rows:
        print("No matching stations found.")
        return 1

    print(f"\n{'WMO':<10}{'VINTAGE':<12}FILENAME")
    for wmo, vint, filename in sorted(set(rows)):
        print(f"{wmo:<10}{vint:<12}{filename}")
    print(f"\n{len(set(rows))} match(es). Put the chosen WMO in climate_config.yaml.")
    return 0


def select_common_vintage(
    available: dict[str, dict[str, str]],
    preferred: list[str],
) -> str:
    """Pick the newest preferred vintage offered by every station."""
    for vintage in preferred:
        missing = [key for key, vintages in available.items() if vintage not in vintages]
        if not missing:
            return vintage
        print(f"   vintage {vintage}: unavailable for {', '.join(sorted(missing))}")

    # Fall back to any vintage common to all stations.
    common = set.intersection(*(set(v) for v in available.values())) if available else set()
    common.discard("all-years")
    if common:
        chosen = sorted(common)[-1]
        print(f"   no preferred vintage is universal; falling back to {chosen}")
        return chosen

    raise SystemExit(
        "\nFATAL: no single TMYx vintage is available for all stations.\n"
        "Mixing vintages would reintroduce the confound this set was rebuilt to\n"
        "remove. Either drop the offending station or accept an older vintage\n"
        "via --vintage. Availability matrix:\n"
        + "\n".join(f"  {k:16s} {sorted(v)}" for k, v in sorted(available.items()))
    )


# ---------------------------------------------------------------------------
# Download and validation
# ---------------------------------------------------------------------------

def validate_epw(text: str, wmo: str) -> tuple[int, str]:
    """Check structure. Returns (n_data_rows, location_line)."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].upper().startswith("LOCATION"):
        raise ValueError("first line is not a LOCATION header")

    n_rows = len(lines) - EPW_HEADER_LINES
    if n_rows != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} data rows, found {n_rows}")

    if wmo.lstrip("0") not in lines[0].replace(" ", ""):
        print(f"      note: WMO {wmo} not echoed in LOCATION line (upstream quirk, not fatal)")

    return n_rows, lines[0]


def download_climate(key: str, entry: dict, url: str, force: bool) -> dict:
    """Download one station's zip, extract the EPW and STAT, validate, record."""
    epw_target = EPW_DIR / entry["epw_filename"]
    stat_target = STAT_DIR / (Path(entry["epw_filename"]).stem + ".stat")

    if epw_target.exists() and not force:
        print(f"   {key:16s} already present, skipping (use --force to replace)")
        return {"key": key, "status": "skipped"}

    print(f"   {key:16s} downloading {url.split('/')[-1]}")
    payload = _fetch(url)

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        epw_names = [n for n in names if n.lower().endswith(".epw")]
        stat_names = [n for n in names if n.lower().endswith(".stat")]
        if not epw_names:
            raise RuntimeError(f"no .epw inside {url}")

        epw_bytes = archive.read(epw_names[0])
        stat_bytes = archive.read(stat_names[0]) if stat_names else b""

    text = epw_bytes.decode("latin-1")
    n_rows, location_line = validate_epw(text, entry["wmo"])

    EPW_DIR.mkdir(parents=True, exist_ok=True)
    STAT_DIR.mkdir(parents=True, exist_ok=True)
    epw_target.write_bytes(epw_bytes)
    if stat_bytes:
        stat_target.write_bytes(stat_bytes)

    print(f"   {' ':16s} -> {epw_target.name}  ({n_rows} rows, {len(epw_bytes) / 1024:.0f} KB)")
    print(f"   {' ':16s}    {location_line[:96]}")

    return {
        "key": key,
        "status": "downloaded",
        "city": entry["city"],
        "role": entry["role"],
        "ashrae_zone": entry["ashrae_zone"],
        "wmo": entry["wmo"],
        "source_url": url,
        "source_zip": url.split("/")[-1],
        "local_epw": str(epw_target.relative_to(REPO_ROOT)),
        "local_stat": str(stat_target.relative_to(REPO_ROOT)) if stat_bytes else None,
        "location_header": location_line,
        "n_data_rows": n_rows,
        "bytes": len(epw_bytes),
        "sha256": hashlib.sha256(epw_bytes).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Download the Phase 2.3 EPW set.")
    parser.add_argument("--vintage", help="force a specific TMYx vintage, e.g. 2009-2023")
    parser.add_argument("--only", help="download a single climate key")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--dry-run", action="store_true", help="probe and report, download nothing")
    parser.add_argument("--list-dir", metavar="KEY",
                        help="list all TMYx stations in this climate's upstream directory, then exit")
    parser.add_argument("--grep", metavar="TEXT",
                        help="with --list-dir, keep only filenames containing TEXT")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    preferred = config["metadata"]["preferred_vintages"]

    if args.list_dir:
        entry = config["climates"].get(args.list_dir)
        if entry is None:
            print(f"Unknown climate key: {args.list_dir}")
            print(f"Known keys: {', '.join(config['climates'])}")
            return 1
        return list_directory(entry["onebuilding_dir"], args.grep, args.vintage)

    climates = {
        key: entry
        for key, entry in config["climates"].items()
        if entry.get("role") != "excluded" and (args.only is None or key == args.only)
    }
    if not climates:
        print(f"No climates selected (--only {args.only}?)")
        return 1

    print(f"\nPhase 2.3 — EPW download")
    print(f"Interpreter: {sys.executable}")
    print(f"Config: {CONFIG_PATH.relative_to(REPO_ROOT)}")
    print(f"Stations: {len(climates)} "
          f"({sum(1 for e in climates.values() if e['role'] == 'training')} training, "
          f"{sum(1 for e in climates.values() if e['role'] == 'holdout')} holdout)\n")

    # ---- Pass 1: probe availability -------------------------------------
    print("Probing available vintages...")
    available: dict[str, dict[str, str]] = {}
    for key, entry in climates.items():
        vintages = list_station_vintages(entry["onebuilding_dir"], entry["wmo"])
        if not vintages:
            print(f"   {key:16s} NO FILES FOUND for WMO {entry['wmo']} in "
                  f"{entry['onebuilding_dir']}")
        else:
            labels = sorted(v for v in vintages if v != "all-years")
            print(f"   {key:16s} {', '.join(labels) or 'all-years only'}")
        available[key] = vintages

    empty = [k for k, v in available.items() if not v]
    if empty:
        print(f"\nFATAL: no files resolved for: {', '.join(empty)}")
        print("Check 'wmo' and 'onebuilding_dir' for those entries in climate_config.yaml.")
        return 1

    vintage = args.vintage or select_common_vintage(available, preferred)
    print(f"\nSelected vintage: TMYx.{vintage}  (applied uniformly to all stations)\n")

    if args.dry_run:
        print("Dry run — nothing downloaded.")
        return 0

    # ---- Pass 2: download ------------------------------------------------
    print("Downloading...")
    records, failures = [], []
    for key, entry in climates.items():
        url = available[key].get(vintage)
        if url is None:
            failures.append((key, f"vintage {vintage} unavailable"))
            print(f"   {key:16s} SKIPPED — vintage {vintage} unavailable")
            continue
        try:
            records.append(download_climate(key, entry, url, args.force))
        except Exception as exc:                       # noqa: BLE001
            failures.append((key, str(exc)))
            print(f"   {key:16s} FAILED — {exc}")

    # ---- Provenance ------------------------------------------------------
    downloaded = [r for r in records if r["status"] == "downloaded"]
    if downloaded:
        existing = {}
        if PROVENANCE_PATH.exists():
            existing = {r["key"]: r for r in json.loads(PROVENANCE_PATH.read_text())["files"]}
        for record in downloaded:
            existing[record["key"]] = record

        PROVENANCE_PATH.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": BASE_URL,
            "vintage": f"TMYx.{vintage}",
            "n_files": len(existing),
            "files": sorted(existing.values(), key=lambda r: r["key"]),
        }, indent=2) + "\n")
        print(f"\nProvenance written: {PROVENANCE_PATH.relative_to(REPO_ROOT)}")

    print(f"\nSummary: {len(downloaded)} downloaded, "
          f"{sum(1 for r in records if r['status'] == 'skipped')} skipped, "
          f"{len(failures)} failed")
    if failures:
        for key, reason in failures:
            print(f"   {key}: {reason}")
        return 1

    print("\nNext: Sub-task 2.3.2 — implement climate/epw_parser.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
