"""
Fear & Greed index — fetched from api.alternative.me (free, no auth).
Value: 0 = extreme fear, 100 = extreme greed.
Cached to disk for 1 hour so the bot doesn't hammer the API.
"""
import json
import os
import time
import urllib.request

_HERE     = os.path.dirname(os.path.abspath(__file__))
_FG_CACHE = os.path.join(_HERE, "data", "fg_cache.json")
_TTL      = 3600  # seconds


def get_current_fg() -> int:
    if os.path.exists(_FG_CACHE):
        try:
            with open(_FG_CACHE) as f:
                c = json.load(f)
            if time.time() - c["fetched_at"] < _TTL:
                return int(c["value"])
        except Exception:
            pass
    try:
        with urllib.request.urlopen(
            "https://api.alternative.me/fng/?limit=1", timeout=10
        ) as r:
            data = json.loads(r.read())
        value = int(data["data"][0]["value"])
        os.makedirs(os.path.dirname(_FG_CACHE), exist_ok=True)
        with open(_FG_CACHE, "w") as f:
            json.dump({"value": value, "fetched_at": time.time()}, f)
        return value
    except Exception:
        return 50  # neutral fallback if API is down


def fetch_fg_history(days: int = 100) -> dict:
    """Return {YYYY-MM-DD: int} for the last N days (UTC dates)."""
    try:
        url = f"https://api.alternative.me/fng/?limit={days}"
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        return {
            time.strftime("%Y-%m-%d", time.gmtime(int(e["timestamp"]))): int(e["value"])
            for e in data["data"]
        }
    except Exception:
        return {}
