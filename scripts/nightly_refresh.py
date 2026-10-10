"""Nightly job: refresh catalog JSON and reload a running API if present.

Cart-history extracts are expensive (4h, $1.21–$4.53), so this job does NOT
pull a new cart export. It only rebuilds sellability files from the latest
local extract and hits POST /v1/admin/reload-rules.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

from generate_catalog_lists import main as generate_catalog  # noqa: E402


def reload_api(base_url: str) -> dict:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/admin/reload-rules",
        method="POST",
        data=b"",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cart-data", default="data/cart_export_19_05.csv")
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument("--skip-reload", action="store_true")
    args = parser.parse_args()

    sys.argv = ["generate_catalog_lists.py", "--cart-data", args.cart_data, "--out-dir", "catalog"]
    generate_catalog()

    if args.skip_reload:
        return
    try:
        payload = reload_api(args.api_url)
        print("API rules reloaded:", payload)
    except OSError as exc:
        print(f"Catalog files updated; API reload skipped ({exc})")


if __name__ == "__main__":
    main()
