#!/usr/bin/env python3
"""
download_fpa_cases.py
---------------------
Downloads FPA Financial Planning Competition (FPC) case study PDFs.

NOTE: These PDFs are copyright of the Financial Planning Association.
      Do NOT commit them to a public repository.
      The raw_external/ folder is in .gitignore.

Usage:
    python scripts/download_fpa_cases.py
"""

import os
import pathlib
import urllib.request
import urllib.error
import sys

# ── Destination ───────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent
RAW_EXT = REPO_ROOT / "streamlit_app" / "data" / "cases" / "raw_external"

# ── PDF catalogue ─────────────────────────────────────────────────────────────
PDFS = [
    (
        "FPA_FPC_2025.pdf",
        "https://www.financialplanningassociation.org/sites/default/files/2025-02/FPC25%20Case%20Study.pdf",
        2025,
    ),
    (
        "FPA_FPC_2024.pdf",
        "https://www.financialplanningassociation.org/sites/default/files/2025-02/FPC24%20Case%20Study%20Final%20Copy.pdf",
        2024,
    ),
    (
        "FPA_FPC_2023.pdf",
        "https://www.financialplanningassociation.org/sites/default/files/2023-02/FPC23%20Case%20Study%20-%20The%20Rogers%20Retire_FINAL.pdf",
        2023,
    ),
    (
        "FPA_FPC_2022.pdf",
        "https://www.financialplanningassociation.org/sites/default/files/2022-02/2022%20FPC%20Case%20Study%20Final.pdf",
        2022,
    ),
    (
        "FPA_FPC_2021.pdf",
        "https://www.financialplanningassociation.org/sites/default/files/2021-02/FPC21%20Case%20Study%20Final.pdf",
        2021,
    ),
    (
        "FPA_FPC_2020.pdf",
        "https://www.financialplanningassociation.org/sites/default/files/2020-05/FPC20%20Case%20Study.pdf",
        2020,
    ),
    (
        "FPA_FPC_2019.pdf",
        "https://www.financialplanningassociation.org/sites/default/files/2021-07/FPC%202019%20Case%20Study.pdf",
        2019,
    ),
    (
        "FPA_FPC_2018.pdf",
        "https://www.financialplanningassociation.org/sites/default/files/2021-07/FPC%202018%20Case%20Study.pdf",
        2018,
    ),
    (
        "FPA_FPC_2016.pdf",
        "https://www.financialplanningassociation.org/sites/default/files/2021-07/FPC%202016%20Case%20Study.pdf",
        2016,
    ),
    (
        "FPA_FPC_2015.pdf",
        "https://www.financialplanningassociation.org/sites/default/files/2021-07/FPC15%20Case%20Study.pdf",
        2015,
    ),
    (
        "FPA_FPC_2014.pdf",
        "https://www.financialplanningassociation.org/sites/default/files/2021-07/FPC%202014%20Case%20Study.pdf",
        2014,
    ),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def download_all(overwrite: bool = False) -> list[str]:
    RAW_EXT.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename, url, year in PDFS:
        dest = RAW_EXT / filename
        if dest.exists() and not overwrite:
            print(f"  ✓ Already exists: {filename}")
            downloaded.append(str(dest))
            continue
        print(f"  ↓ Downloading FPC {year}: {filename} ...", end="", flush=True)
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
            dest.write_bytes(content)
            kb = len(content) // 1024
            print(f" {kb} KB ✓")
            downloaded.append(str(dest))
        except urllib.error.HTTPError as e:
            print(f" HTTP {e.code} — skipped")
        except urllib.error.URLError as e:
            print(f" Network error: {e.reason} — skipped")
        except Exception as e:
            print(f" Error: {e} — skipped")
    return downloaded


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download FPA FPC case study PDFs")
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    print(f"\nDownloading {len(PDFS)} FPA FPC case study PDFs → {RAW_EXT}\n")
    paths = download_all(overwrite=args.overwrite)
    found = [p for p in paths if os.path.exists(p)]
    print(f"\nDone. {len(found)}/{len(PDFS)} PDFs available in raw_external/")
    sys.exit(0)
