# Certa examples

Two deployment options, one interface — see the **Deployment** section of the
[main README](../README.md). `decide_local.py` runs the model in-process; `backends.py` picks
local vs a remote gateway from the environment with identical calling code.

## 1. Local decision (no server)
```bash
pip install -e .
python examples/decide_local.py                 # uses the released model
python examples/decide_local.py runs/rl/best     # or a local checkpoint
```

## 1b. Choose backend by environment (local or gateway)
```bash
# local (in-process):
CERTA_BACKEND=local CERTA_CHECKPOINT=goutam/LFM2.5-1.2B-RLCD python examples/backends.py
# remote (a gateway on another box):
CERTA_BACKEND=remote CERTA_BASE_URL=http://gateway:8000 python examples/backends.py
```

## 2. Run the server
```bash
pip install -e '.[serve]'
CERTA_CHECKPOINT=goutam/LFM2.5-1.2B-RLCD python -m certa --host 0.0.0.0 --port 8000
# optional: require auth with  CERTA_API_KEY=secret
```

### Call it with curl (Jev-compatible wire protocol)
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

### Drop-in with the official `typesafe-sdk`
Certa serves Jev's wire protocol, so an existing Jev client just repoints its base URL:
```python
from typesafe_sdk import TypeSafeClient, Choice
client = TypeSafeClient(base_url="http://localhost:8000", api_key="secret")
resp = client.system_one(state="...", questions={"team": Choice(instructions="route", criteria={...})})
print(resp.answers["team"].choice, resp.answers["team"].confidence)
```

### Or Certa's own client (extra: `certa[client]`)
```python
from certa.client import Client, choice
with Client("http://localhost:8000", api_key="secret") as c:
    r = c.decide("...", {"team": choice("route", {"billing":"...","technical":"..."})})
    print(r.answers["team"].choice)
```
