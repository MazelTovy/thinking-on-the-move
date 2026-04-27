#!/usr/bin/env python3
"""
Download Advan Weekly Patterns Plus (NO NAICS filter) for 2021-2022.
Then filter locally to NAICS 722% (all food services).
~104 files, ~16 GB total.
"""

import json
import os
import subprocess
import time

API_URL = "https://api.deweydata.io/api/v1/external/data/prj_ubodibfx__cdst_89oe9nc8ijfpc96v"
API_KEY = "akv1_fHfz-4AVTTI5ZrT0YPlcEa7cG842cufruPY"

OUT_DIR = "/scratch/sx2490/econai/nyc_metro/data/advan/wpp_full_raw"
os.makedirs(OUT_DIR, exist_ok=True)

PAGES_TO_SCAN = [5, 6, 7]
DATE_START = "2021-01-01"
DATE_END = "2022-12-31"


def fetch_page(page):
    result = subprocess.run(
        ["curl", "-s", "-H", f"X-API-KEY: {API_KEY}", f"{API_URL}?page={page}"],
        capture_output=True, text=True, timeout=60
    )
    return json.loads(result.stdout)


def download_file(link, out_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["curl", "-s", "-o", out_path, link],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
                return True
            raise RuntimeError(f"curl exit {result.returncode}")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt+1}: {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"    FAILED: {e}")
                return False


def main():
    targets = []
    for page in PAGES_TO_SCAN:
        print(f"Scanning page {page}...")
        data = fetch_page(page)
        for item in data["download_links"]:
            pk = item["partition_key"]
            if DATE_START <= pk <= DATE_END:
                targets.append(item)
        print(f"  Found {len(targets)} files so far")

    total_gb = sum(t["file_size_bytes"] for t in targets) / 1024**3
    print(f"\nTotal: {len(targets)} files, {total_gb:.1f} GB")

    downloaded = skipped = failed = 0
    for i, item in enumerate(targets):
        pk = item["partition_key"]
        fname = f"wpp_full_{pk}.parquet"
        out_path = os.path.join(OUT_DIR, fname)

        if os.path.exists(out_path):
            existing = os.path.getsize(out_path)
            expected = item["file_size_bytes"]
            if abs(existing - expected) < 10240:
                skipped += 1
                continue

        size_mb = item["file_size_bytes"] / 1024 / 1024
        print(f"[{i+1}/{len(targets)}] {pk} ({size_mb:.0f} MB)...", end=" ", flush=True)
        if download_file(item["link"], out_path):
            downloaded += 1
            print("OK")
        else:
            failed += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
