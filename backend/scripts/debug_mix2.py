"""
debug_mix2.py  —  prints raw Mubasher mix2 response so we can fix the parser
"""
import json
import requests
from datetime import datetime, timezone
from urllib.parse import urlencode

BASE = "https://data-sa9.mubasher.net/mix2"

# Build a minimal URL — same params as the fetch script
params = {
    "SID": "sid",
    "UID": "123",
    "RT":  27,           # news
    "E":   "TDWL",
    "UE":  "TDWL",
    "L":   "EN",
    "AE":  1,
    "UNC": 0,
    "M":   1,
    "H":   1,
    "SD":  "20250101000000",
    "ED":  "20251108000000",
}

url = f"{BASE}?{urlencode(params)}"
print("URL:", url)
print()

r = requests.get(url, timeout=30)
print("HTTP Status:", r.status_code)
print("Content-Type:", r.headers.get("content-type"))
print()

# Try to parse JSON
try:
    data = r.json()
    print("=== FULL JSON (first 3000 chars) ===")
    print(json.dumps(data, indent=2)[:3000])
    print()
    print("=== TOP-LEVEL KEYS ===", list(data.keys()))
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"  {k} -> keys: {list(v.keys())}")
        elif isinstance(v, list):
            print(f"  {k} -> list len={len(v)}, first item: {str(v[0])[:120] if v else 'empty'}")
        else:
            print(f"  {k} -> {str(v)[:120]}")
except Exception as e:
    print("JSON parse failed:", e)
    print("Raw text (first 2000):", r.text[:2000])
