# On-chain risk ML: when a fast model should not ship

Can the shape of a Bitcoin transaction's **already-observed ancestry** prioritize an analyst's review queue? On the Elliptic benchmark, gradient boosting ranked illicit labels better than logistic regression **before** calibration. A later sigmoid calibration reversed the ranking. At the frozen review threshold, the calibrated boosted model caught 825 of 828 labelled illicit transactions but sent **13,101 of 13,123** labelled transactions to review. **Do not deploy this policy.** This repo serves the failed experiment so the result, including its operational cost, can be reproduced.

[Evaluation with per-step counts](results/evaluation.json) · [HTTP benchmark](results/latency.json) · [Design and leakage contract](DESIGN.md)

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). No credentials, exchange account, chain RPC or paid API. From the repo root:

```sh
uv sync --locked --python 3.12
uv --offline run pytest               # synthetic contract tests; no dataset needed
uv run python -m risk.data             # one-time public download: ~697 MB, verifies SHA-256
uv --offline run python -m risk.train # writes artifacts/ locally and results/evaluation.json
OMP_NUM_THREADS=1 uv --offline run uvicorn risk.api:create_app --factory --host 127.0.0.1 --port 8000
```

In a second terminal:

```sh
# Invented context to demonstrate the HTTP contract, NOT a real transaction.
curl -sS http://127.0.0.1:8000/score -H 'Content-Type: application/json' \
  -d '{"parents":[[0,0],[1,1]],"prior_mean":1.0}'
uv --offline run python -m risk.bench
curl -sS http://127.0.0.1:8000/drift
```

Each parent pair is `[observed incoming transaction count, longest observed upstream path in edges]`. `prior_mean` is total incoming edges / total nodes in the preceding three completed time steps; use zero when none exist. Build this context in a trusted upstream indexer. The service **does not** look up addresses or verify client-supplied context. Unknown fields, negative counts and non-finite numbers are rejected. JSON `probability` is a research model output, not a validated probability of illegal activity.

## Data and availability

[Elliptic's dataset card](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and [Weber et al., KDD Workshop on Anomaly Detection in Finance, 2019](https://arxiv.org/abs/1908.02591). Each node is a transaction; each directed edge represents a Bitcoin flow. The 49 time steps are sampled components about two weeks apart, **not** continuous blockchain history. Graph roots can simply be parents missing from the sample. Positive = transaction associated with an entity Elliptic labelled illicit, not an independent finding about a person.

The [pinned public mirror](https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/tree/d075df05ace1db9a826512fe71806aaecb76f97e) provided all three CSVs without login. [Download code and hashes](risk/data.py). The mirror's MIT tag conflicts with the publisher: follow Elliptic's **[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)** for the data. This repository distributes original code under MIT and aggregate evaluation results, not the raw data, per-transaction examples or trained artifact. Check licensing before any reuse beyond research.

**203,769 nodes, 234,355 edges, 157,205 unknown labels.** Unknowns participate in graph and historical traffic features but are never treated as licit. Of 46,564 labelled nodes, 4,545 are illicit. Label ascertainment dates aren't public; the split enforces feature chronology but cannot reproduce historical label maturity. Test-set prevalence is not Bitcoin-wide prevalence.

## What was known at scoring time?

Eight features: observed input count, longest upstream path, mean/max parent input count, mean parent depth, fraction of parents that are observed roots, three-step historical mean input count, and current count minus that mean. The builder follows the directed acyclic graph, not CSV order or numeric IDs. Same-step parents are observable because their outputs must exist before the child spends them. The temporal reference excludes the current step. The model never sees future spending, absolute step, entity labels, or Elliptic's anonymous local/forward-neighborhood aggregates. This sacrifices useful signal in exchange for a feature boundary that can be audited.

| Steps | Purpose | Licit | Illicit | Unknown | Illicit among labelled |
|---|---|---:|---:|---:|---:|
| 1–27 | Train logistic and boosted trees | 22,466 | 2,457 | 89,953 | 9.86% |
| 28–32 | Fit separate sigmoid per model on raw margins | 3,070 | 945 | 11,737 | 23.54% |
| 33–37 | Choose each model's threshold | 4,188 | 315 | 16,340 | 7.00% |
| 38–49 | Evaluate once per fixed specification | 12,295 | 828 | 39,175 | 6.31% |

Logistic regression uses training-only standardization. Boosting uses 150 histogram trees, at most 15 leaves each, no random validation split or early stopping. The policy uses the **highest score cutoff recovering at least 80% of policy-window positives**, including every tied score. No test labels select thresholds. Tests append descendants and flip every known holdout label to check earlier features, fitted scores, and thresholds remain unchanged.

## Held-out results: real Elliptic labels

| Model | AP raw → calibrated | Brier raw → calibrated ↓ | Frozen precision | Frozen recall | Flagged | Retrospective P at ≥80% recall |
|---|---:|---:|---:|---:|---:|---:|
| Logistic | .0797 → .0518 | .0616 → .0895 | 5.65% | 70.65% | 10,348 / 13,123 | 6.06% |
| Histogram gradient boosting | .1416 → .0410 | .0598 → .0927 | 6.30% | 99.64% | 13,101 / 13,123 | 5.48% |

AP is average precision; a constant score has AP equal to prevalence (6.31%). The final column is a *descriptive* test PR operating point, computed with the same highest-cutoff/tie rule; it is not the policy served. At the boosted validation threshold of **0.209515**, the test confusion counts are **825 TP, 12,276 FP, 19 TN, 3 FN**. A desk reviewing 100 flagged labelled cases would find about six positives here, no better than reviewing 100 at random. Outcomes for the unknowns cannot be quantified.

**Why ranking got worse:** the calibration window induced negative sigmoid slopes (boosting −0.0665, logistic −0.0106), reversing the base models' rankings. Its 23.54% positive rate also differed sharply from the test window's 6.31%. Brier scores got worse, not better. At the frozen cutoff, observed policy-window recall already exceeded 80% because of ties. Test positive counts varied from **239 in step 42 to 2 in step 46**; a single aggregate obscures that volatility. The raw boosted ranking is a hypothesis worth testing, not a new winner chosen on this holdout. Before reconsidering: compare monotonic/intercept-only calibration and a fixed analyst-capacity policy in walk-forward development folds, then evaluate a locked decision on *new* data with mature labels.

## Serving, drift and latency

On Apple M4 / macOS 27.2.1, Python 3.12.15: **p50 0.98 ms, p95 2.54 ms, p99 3.72 ms** over 1,024 serial warm loopback HTTP POSTs. One Uvicorn worker, persistent connection, 20 warmups, 256 deterministically sampled labelled held-out rows replayed four times, LLM off. All 1,024 responses matched batch scoring within numerical tolerance. This includes validation, feature computation from supplied context, model inference, drift-window insertion and the HTTP round trip. It excludes upstream graph indexing, TLS, cold starts and contention. Not a production SLO.

`GET /drift` compares the last 512 request feature vectors to **all** training traffic (including unlabeled nodes). It returns `insufficient` below 64 rows, then per-feature population stability index using fixed training quantile bins, explicit tails and 0.5-count smoothing. It returns `alert` at PSI ≥0.25 on any feature. Across all 52,298 real test-window nodes, `recent_mean_inputs` PSI was **6.47** and `depth` **0.264**. The historic mean repeats across a whole time step, so only a handful of independent time periods drive that figure. An alert calls for investigation, not a finding of fraud or proof of performance loss. Online windows are in-memory, per process and reset on restart. A repeated benchmark request is counted repeatedly. Use one worker for this demo; aggregate telemetry outside workers in a real service.

## Optional local LLM: choose evidence, never a verdict

Disabled unless the server has `RISK_LOCAL_LLM` **and** the caller supplies `?explain=true`. The LLM can choose one of three numerical observations; it cannot supply arbitrary prose or modify a score. The application renders the selected observation from the actual features. This is descriptive context, not causal feature attribution. During development free-text generation called a graph a tree without enough evidence; the allowlist removes that failure mode. A malformed choice, timeout or unavailable local model returns `explanation.status=unavailable` and leaves the score intact.

Optional Apple Silicon demo (initial free model/dependency download; not needed for testing or scoring):

```sh
uv run --no-project --isolated --python 3.12 --with mlx-lm==0.32.2 mlx_lm.server \
  --model mlx-community/SmolLM2-360M-Instruct-6bit --host 127.0.0.1 --port 18492
# Restart the scoring process in another terminal:
RISK_LOCAL_LLM=mlx-community/SmolLM2-360M-Instruct-6bit OMP_NUM_THREADS=1 \
  uv --offline run uvicorn risk.api:create_app --factory --host 127.0.0.1 --port 8000
curl -sS 'http://127.0.0.1:8000/score?explain=true' -H 'Content-Type: application/json' \
  -d '{"parents":[[0,0],[1,1]],"prior_mean":1.0}'
```

Observed on that **synthetic** input with real local inference: same `probability=0.267161913478384` and `review=true` with or without explanation; selected text: “Observed parents have mean input count 0.5 and maximum 1.” No model quality or broader explanation-utility claim follows from one example. Bind both servers to loopback. Never load untrusted joblib files; deserialization can execute code. No authentication, enforcement or durable per-case logging is provided.

## Two-minute walkthrough

- **0:00–0:25:** Open the confusion counts. Ask how useful 99.64% recall is when only 22 labelled cases escape review. State the no-deploy decision.
- **0:25–0:55:** Follow one child to its parents in `risk/graph.py`. Point out why outgoing edges and current-step totals would leak future information. Show the four chronological windows and why unknowns aren't negatives.
- **0:55–1:20:** Compare raw and calibrated AP and Brier. Show the negative slope and calibration prevalence; distinguish the validation-selected operating point from hindsight precision at fixed recall.
- **1:20–1:45:** POST the synthetic context, run `python -m risk.bench` via uv, open `/drift`. Distinguish the real HTTP timing from graph lookup or concurrent production traffic.
- **1:45–2:00:** Opt in to local evidence selection, compare the unchanged score, and end with the next decision: evaluate a constrained policy on a new time window, not this holdout.
