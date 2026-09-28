# Certa

**Calibrated, typed decisions from a small open-weight model.**

Ask typed questions about any text and get back typed answers — each with a **probability you
can actually trust**. No text generation, no JSON to parse, no prompt engineering. Just
`pip install`, then one function call.

## Quickstart

```bash
pip install certa
```

```python
from certa.engine import ModelEngine
from certa.client import choice, score, verify

engine = ModelEngine("goutam/LFM2.5-1.2B-RLCD")   # downloads the model once

resp = engine.decide(
    "I was double-charged and support has ignored me for three days.",
    {
        "team":   choice("Route this ticket", {"billing": "charges", "technical": "bugs", "sales": "pricing"}),
        "anger":  score("How angry is the customer?", ["calm", "annoyed", "furious"]),
        "urgent": verify("Does the message convey urgency?"),
    },
)

resp.answers["team"].choice          # 'billing'
resp.answers["team"].confidence      # 0.79
resp.answers["team"].probabilities   # {'billing': 0.79, 'technical': 0.04, 'sales': 0.17}
resp.answers["anger"].score          # 1.44   (probability-weighted level, 0–2)
resp.answers["urgent"].noul          # 0.78   (P(true))
```

That's the whole API for a decision — **`engine.decide(state, questions)`**. Each answer is a
typed value plus a calibrated confidence you can threshold: act on the confident ones, escalate
the rest. Runs in ~14 ms (warm) on a GPU, and no server is required to get started.

## Why Certa

- **Typed, not textual.** Answers are structured values with probabilities — no JSON-from-prose, no regex, no retries.
- **Calibrated by construction.** Trained with a proper-scoring objective, so a reported 0.8 means right ≈ 80% of the time (in-distribution ECE 0.043).
- **Fast and flat.** Every question in a request is scored in one batched forward pass, so latency barely grows with more questions.
- **Drop-in for [Jev](https://docs.typesafe.ai/).** Same `POST /v1/systemone` wire protocol and a `typesafe-sdk`-compatible client — repoint an existing Jev client and it just works.
- **Open weights, self-hosted.** No per-call API cost, full data control.

## Install

```bash
pip install certa            # core: decision engine (what the Quickstart uses)
pip install 'certa[serve]'   # + FastAPI server, to run it as a service
pip install 'certa[client]'  # + HTTP client for calling a remote server
```

Needs Python ≥ 3.10 and PyTorch; a CUDA GPU is recommended for low latency (it runs on CPU too,
just slower).

## Primitives

| Type | Ask | Returns |
|------|-----|---------|
| **Choice** | pick one option from an unordered set | `choice`, `probabilities`, `confidence` |
| **Score** | place on an ordered scale | `score` (weighted mean of level indices), `legend`, `probabilities`, `confidence` |
| **Noul** | is a statement true? | `noul` — P(true) ∈ [0, 1] (no confidence field) |

Confidence matches Jev exactly: `confidence = (n·p_max − 1) / (n − 1)`, clamped to `[0, 1]`
(0 = uniform, 1 = all mass on one option). The wire protocol accepts Jev's limit of 255 Choice
options; the current model's single-letter decode supports **≤ 26 options** (larger sets return
`422`).

## Serve it

```bash
CERTA_CHECKPOINT=goutam/LFM2.5-1.2B-RLCD python -m certa --port 8000
# optional bearer auth:  CERTA_API_KEY=secret python -m certa --port 8000
```

```bash
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "I was double-charged and support ignored me for days.",
  "questions": { "team": {"type":"choice","instructions":"route",
    "criteria":{"billing":"charges","technical":"bugs","sales":"pricing"}} }
}'
```

An existing Jev client works unchanged — just repoint `base_url`:

```python
from typesafe_sdk import TypeSafeClient, Choice
client = TypeSafeClient(base_url="http://localhost:8000")
client.system_one(state="...", questions={"team": Choice(instructions="route", criteria={...})})
```

More in [`examples/`](examples/).

## Deployment: local or gateway

Every backend implements one interface — `decide(state, questions) → response` — so *where the
model runs is configuration, not a code change*. Start local, move to a gateway later with zero
edits to your calling code.

```python
from certa.backend import LocalBackend, RemoteBackend, from_env

backend = LocalBackend("goutam/LFM2.5-1.2B-RLCD")          # A) model in this process
backend = RemoteBackend("http://gateway:8000", api_key="…") # B) call a gateway on another box
backend = from_env()                                        # or choose via environment
resp = backend.decide(state, questions)                     # identical call in all cases
```

**A. Local (in-process).** Model and caller on one machine — lowest latency, no network hop.
Best for a single service that owns a GPU, or for embedding decisions inside a larger app.

**B. Gateway (remote).** Run the model on one GPU box and point many light clients at it. The
gateway is the built-in FastAPI server; it adds **cross-request micro-batching** (coalesces
concurrent requests into one forward pass — the main throughput lever for this prefill-bound
workload) and an optional **result cache** (safe because decisions are deterministic). Measured
on an L40S: single-request latency ≈ 14 ms; micro-batching lifts 32 concurrent requests from
**≈ 69 → ≈ 403 req/s (5.8×)**.

```bash
# on the gateway (GPU box):
CERTA_BACKEND=local CERTA_CHECKPOINT=goutam/LFM2.5-1.2B-RLCD \
CERTA_BATCH=1 CERTA_CACHE=1 CERTA_API_KEY=secret \
  python -m certa --host 0.0.0.0 --port 8000

# on each client:
CERTA_BACKEND=remote CERTA_BASE_URL=http://gateway:8000 CERTA_API_KEY=secret python your_app.py
```

| Env var | Meaning |
|---|---|
| `CERTA_BACKEND` | `local` (default) · `remote` · `vllm` |
| `CERTA_CHECKPOINT` | model to load (local backend) |
| `CERTA_BASE_URL` | gateway URL (remote backend) |
| `CERTA_API_KEY` | bearer token (set on gateway to require it; on client to send it) |
| `CERTA_BATCH` | `1` to enable cross-request micro-batching on the gateway |
| `CERTA_CACHE` | `1` to cache identical requests |

Scale the gateway horizontally by running replicas behind a load balancer — it's stateless
(each decision is a single pass, no cross-request state). **vLLM** is available as an opt-in
`vllm` backend, but it targets long autoregressive generation; for Certa's single-pass decode
the batched local/gateway path is simpler and preserves exact calibrated semantics — reserve
vLLM for the generative "System-Two" model you escalate *to*.

## Performance & model

| | In-distribution (test) | Warm latency (L40S) |
|---|---|---|
| **[`goutam/LFM2.5-1.2B-RLCD`](https://huggingface.co/goutam/LFM2.5-1.2B-RLCD)** | 0.766 acc · 0.319 Brier · 0.043 ECE · 0.079 AURC | ≈ 14 ms / decision |

Base `LiquidAI/LFM2.5-1.2B-Base`, trained via supervised proper-scoring + RLCD refinement. Full
recipe, per-split metrics, and limitations (notably out-of-distribution behaviour) are in the
[model card](https://huggingface.co/goutam/LFM2.5-1.2B-RLCD).

## Architecture

```
certa/
  contract/   # the decode contract (schemas, encoding, confidence)
  engine.py   # single forward pass → per-option probabilities
  server.py   # FastAPI: POST /v1/systemone, GET /v1/models, bearer auth
  client.py   # typesafe-sdk-compatible client + choice()/score()/verify() builders
```

`certa.contract` defines how a question is encoded, scored, and answered. The training pipeline
re-implements the same format independently; a shared format-pinning test in both keeps the
trained model and the server in agreement.

## Development

```bash
pip install -e '.[serve,client,test]'
pytest            # includes real typesafe-sdk wire-parity tests
```

## Security

- **Auth:** the gateway enforces a bearer token when `CERTA_API_KEY` is set, compared in
  **constant time** (`hmac.compare_digest`). No key ⇒ open (intended for trusted networks).
- **Transport:** `RemoteBackend`/`Client` send the key in the `Authorization` header — put the
  gateway behind **HTTPS or a private network**; don't send keys over plain HTTP across the wire.
- **No remote code execution:** models load via **safetensors** (no pickle) and `trust_remote_code`
  is never enabled, so pointing at a checkpoint can't run arbitrary code. Still, only load
  checkpoints you trust.
- **Untrusted input:** the decode is *constrained* — an adversarial `state` can only shift the
  probability distribution over the declared options, never produce free-form output or execute
  anything. Treat the resulting decision like any classifier output on adversarial data.
- **Secrets:** read from environment only; none are hardcoded, logged, or written to disk by the
  library. Request logging is limited to a random `x-certa-request-id` (no bodies, no keys).

## License

This code is licensed under **Apache-2.0** — see [LICENSE](LICENSE). The **model weights are
licensed separately** under the base model's terms (LFM Open License v1.0); see the
[model card](https://huggingface.co/goutam/LFM2.5-1.2B-RLCD) for details.

Not affiliated with TypeSafe.ai; "Jev" is referenced only for API compatibility.
