# Certa examples

Runnable scripts showing the common ways to use Certa. Install first (from the repo root, or
from GitHub):

```bash
pip install -e .                                       # from a clone
# or:  pip install "git+https://github.com/Kurumella/certa.git"
```

## Which entry point should I use?

All of these expose the **same** call — `.decide(state, questions)` — they differ only in *where*
the model runs and how you choose that:

| Entry point | Import | Use it when |
|---|---|---|
| **`ModelEngine`** | `from certa.engine import ModelEngine` | Simplest: run the model **in this process**. Great for scripts and embedding. |
| **`LocalBackend`** | `from certa.backend import LocalBackend` | Same as `ModelEngine` (it just wraps it), but behind the swappable backend interface. |
| **`RemoteBackend`** | `from certa.backend import RemoteBackend` | Call a Certa **gateway on another machine** over HTTP. |
| **`from_env()`** | `from certa.backend import from_env` | Pick local vs remote from env (`CERTA_BACKEND`) — switch deployment **without code changes**. |

Rule of thumb: reach for **`ModelEngine`** for a simple local script; use the **backends** when you
want config-driven local↔remote (or caching). `LocalBackend("…").decide(...)` and
`ModelEngine("…").decide(...)` produce identical results.

## The examples

| File | What it shows |
|---|---|
| [`decide_local.py`](decide_local.py) | The basics — one in-process decision over all three primitives, with warm latency. |
| [`triage_batch.py`](triage_batch.py) | Route a batch of tickets; use **confidence to auto-act vs. escalate**. |
| [`moderation.py`](moderation.py) | A different domain — category + severity + boolean flags in one pass. |
| [`backends.py`](backends.py) | Choose **local vs. remote** by environment via `from_env()`. |
| [`remote_client.py`](remote_client.py) | Call a running **gateway** over HTTP with `certa.client.Client`. |

```bash
python examples/decide_local.py            # or triage_batch.py, moderation.py, ...
python examples/decide_local.py runs/best  # decide_local.py also accepts a local checkpoint path
```

`backends.py` reads the environment:
```bash
CERTA_BACKEND=local  CERTA_CHECKPOINT=goutam/LFM2.5-1.2B-RLCD python examples/backends.py
CERTA_BACKEND=remote CERTA_BASE_URL=http://gateway:8000        python examples/backends.py
```

## Running as a server (gateway)

```bash
pip install "certa[serve] @ git+https://github.com/Kurumella/certa.git"
CERTA_CHECKPOINT=goutam/LFM2.5-1.2B-RLCD python -m certa --host 0.0.0.0 --port 8000
# optional: require auth with  CERTA_API_KEY=secret    | throughput: CERTA_BATCH=1  | cache: CERTA_CACHE=1
```

Call it with **curl** (Jev-compatible wire protocol):
```bash
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "I was double-charged and support ignored me for days.",
  "questions": {
    "team":   {"type":"choice","instructions":"route","criteria":{"billing":"charges","technical":"bugs","sales":"pricing"}},
    "anger":  {"type":"score","instructions":"anger level","criteria":["calm","annoyed","furious"]},
    "urgent": {"type":"noul","instructions":"is it urgent?"}
  }
}'
```

Drop-in with the official **`typesafe-sdk`** — an existing Jev client just repoints its base URL:
```python
from typesafe_sdk import TypeSafeClient, Choice
client = TypeSafeClient(base_url="http://localhost:8000", api_key="secret")
resp = client.system_one(state="...", questions={"team": Choice(instructions="route", criteria={...})})
print(resp.answers["team"].choice, resp.answers["team"].confidence)
```

Or Certa's own client — see [`remote_client.py`](remote_client.py).
