# Scoring contract

Purpose: prioritize transactions for human review, not determine guilt, block funds, or impersonate an exchange. A transaction is observable once its inputs are known. A child can spend only an already-created parent, including within one Elliptic time step; integer node IDs and CSV row order are not clocks.

- Input: observed incoming edges, direct parents' observed in-degree and longest upstream path, and mean in-degree over nodes in the three preceding **completed** time steps. No outputs, current-step totals, labels, absolute time, or provider-normalized anonymous covariates. Unknown-label nodes contribute structure, never negatives. Prior-step reference defaults to zero on cold start. Precomputed summaries are a trusted caller's responsibility, not a public blockchain lookup service. Both training and serving use one feature function.
- Fixed windows: train steps 1–27, calibrate 28–32, choose review threshold 33–37, test 38–49. Fit logistic baseline and histogram gradient boosting without test-tuned knobs. Sigmoid calibration uses untouched raw margins. Choose the highest threshold achieving at least 80% recall on policy data (include ties). Report test precision/recall at that frozen threshold separately from retrospective test precision at 80% recall. No promise of 80% future recall.
- Deployment: train locally, then start a single-worker loopback FastAPI service with the trusted artifact. A rolling drift check compares feature distributions to training bins, never requires labels. Optional local-only LLM produces commentary after scoring; a timeout or missing model must not change the decision. No chain RPC, downstream enforcement, commercial API or real-time investigator feedback.

## Verification contract

| Boundary | Observable check |
|---|---|
| Graph / clock | Appending future nodes or children preserves earlier rows. Known hand-computed ancestry, repeated step and previous-step calculations match. Cycles or impossible edge chronology fail closed. |
| Evaluation | Known threshold with tied scores; disjoint ordered folds; report all denominators, misses, false alarms and per-step variation on actual public labels. |
| Serving | POST a real held-out event through HTTP, compare the score to batch computation; reject malformed context. Benchmark warm serial loopback HTTP, not a Python function. |
| Monitoring / explanations | Too few observations return insufficient data. A shifted synthetic window alerts; stable windows do not. LLM disabled by default, bounded and advisory only. |

No claim about production latency, real-world prevalence, label availability dates, wallet identity, or profitability. Prefer transparent structural signals over the dataset's anonymized features: their scaling and exact availability cannot be independently reconstructed at an event boundary.