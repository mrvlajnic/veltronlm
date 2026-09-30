"""Probe permissively-licensed data sources reachable from this host.

Run before building the acquisition pipeline so every source we later reference is known
to actually respond. Prints status codes and byte sizes only -- no fabricated coverage.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

UA = "VeltronLM/0.1 (research; +https://example.invalid/veltronlm)"
TIMEOUT = 25

PROBES = [
    ("wikimedia-api-en", "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&explaintext=1&exintro=1&titles=Rome"),
    ("wikimedia-api-sr", "https://sr.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&explaintext=1&exintro=1&titles=Rim"),
    ("wikimedia-rest-sr", "https://sr.wikipedia.org/api/rest_v1/page/summary/Belgrade"),
    ("wikisource-gutenberg", "https://en.wikisource.org/w/api.php?action=query&format=json&list=categorymembers&cmtitle=Category:Aesop%27s_Fables&cmlimit=5"),
    ("gutenberg-txt", "https://www.gutenberg.org/files/11/11-0.txt"),
    ("gutenberg-catalog", "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"),
    ("github-raw-mit", "https://raw.githubusercontent.com/pallets/flask/main/LICENSE.txt"),
    ("github-raw-py", "https://raw.githubusercontent.com/python/cpython/main/Lib/os.py"),
    ("github-raw-rust", "https://raw.githubusercontent.com/rust-lang/rust/master/README.md"),
    ("hf-datasets-api", "https://huggingface.co/api/datasets?limit=3"),
    ("hf-resolve-wikitext", "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet"),
    ("pypi", "https://pypi.org/pypi/numpy/json"),
    ("raw-githubusercontent-root", "https://raw.githubusercontent.com/"),
    ("unicode-data", "https://www.unicode.org/Public/UCD/latest/ucd/Names.txt"),
]


def probe(name: str, url: str) -> dict:
    t0 = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-65535"})
    out = {
        "source": name,
        "url": url,
        "status": None,
        "bytes": 0,
        "seconds": 0.0,
        "note": "",
    }
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read(65536)
            out["status"] = resp.status
            out["bytes"] = len(data)
            out["content_type"] = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        out["status"] = e.code
        out["note"] = f"HTTPError {e.reason}"
    except Exception as e:
        out["status"] = -1
        out["note"] = f"{type(e).__name__}: {e}"
    out["seconds"] = round(time.perf_counter() - t0, 2)
    return out


def main() -> int:
    print("=" * 100)
    print("DATA SOURCE REACHABILITY PROBE")
    print("=" * 100)
    results = []
    for name, url in PROBES:
        r = probe(name, url)
        results.append(r)
        flag = "OK  " if (r["status"] and 200 <= r["status"] < 400) else "FAIL"
        print(
            f"{flag} {r['source']:30} status={str(r['status']):>5} "
            f"bytes={r['bytes']:>7} {r['seconds']:>6.2f}s  {r['note']}"
        )

    ok = [r["source"] for r in results if r["status"] and 200 <= r["status"] < 400]
    fail = [r["source"] for r in results if not (r["status"] and 200 <= r["status"] < 400)]
    print()
    print(f"reachable ({len(ok)}): {', '.join(ok)}")
    print(f"unreachable ({len(fail)}): {', '.join(fail) if fail else 'none'}")

    with open("reports/source_probe.json", "w", encoding="utf-8") as fh:
        json.dump({"results": results, "reachable": ok, "unreachable": fail}, fh, indent=2)
    print("wrote reports/source_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
