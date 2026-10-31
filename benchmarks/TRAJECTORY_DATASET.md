# Pramaan trajectory dataset schema

`trajectory_schema_version: "1.1"` records one task as an ordered `calls` array. Each call preserves the exact system/user messages, model request identity and decoding parameters, raw response, provider metadata or bounded error, parsed candidate, deterministic verifier report, and pass/fail label. Call and run telemetry include prompt, cached-prompt, completion, and total tokens; request elapsed time; provider-reported latency; verification time; retry counts; and cost. A cost value is marked either provider-reported or estimated from a dated, sourced model rate card; estimates remain separate from billed spend and account credits. A call without a candidate is retained when a provider request fails. The top-level record links the task to its prompt suite digest, model, validator version, timing, final benchmark outcomes, and data role.

## Data roles and reuse

- `benchmark_evaluation` means a frozen, scored evaluation example. It is **not training eligible**. Do not use it for fine-tuning, prompt tuning, or model selection and then report the same score as held-out performance.
- `training` data comes only from the separately versioned `train-v1.2` prompt suite with its own digest. Keep it distinct from the current held-out `eval-v2.1` suite. The older `train-v1` and `train-v1.1` prompts and their Beam trajectories are superseded and must not be reused as current training data.
- Flatten calls from training trajectories into chat-completion examples as `messages` → `raw_response`, retaining trajectory ID, call index, phase, verifier outcome, and model metadata. Keep repairs linked to the earlier call; avoid treating verifier-provided repair context as an independent user task.
- Keep provider credentials, authorization headers, and raw provider error bodies out of datasets. Request messages and model responses are included because they are required to reproduce training/evaluation examples.

## Public community model runs

The `community` config in `kshitijthakkar/pramaan-proof-of-art-eval-v2.1` is separate from the canonical held-out `compact`, `summary`, and `trajectories` configs. Each public submission is stored under `community/test/<run_id>.jsonl`, tagged `data_role: community_evaluation` and `training_eligible: false`. Rows use the frozen eval-v2.1 suite digest and the same verifier. Historical eval-v2 runs are held in `kshitijthakkar/pramaan-proof-of-art-eval-v2`; they are superseded and must not be ranked or combined with v2.1 runs, even when prompt IDs overlap. Partial runs retain their completed question IDs and coverage; they never update or merge into the official benchmark score. Compare models on their common completed questions and show coverage beside every score.

The Space accepts an API key only for a selected supported provider/model during that user's run. The key is kept in process memory, redacted from provider exceptions, and excluded from local result rows, the Hub dataset, and gallery payloads. The Space uses its server-side `HF_TOKEN` only to publish the sanitized, replay-checked export. Each model run uses a 4,096 output-token ceiling, three-proposal limit, and independent USD 10 reservation cap. Beam and DeepSeek use temperature 0.4; the selected CometAPI DeepSeek alias also sets `extra_body.thinking.type=disabled` because DeepSeek's API enables thinking by default and ignores temperature while thinking is enabled. Anthropic requests omit temperature because the provider rejects it, and use `output_config.effort=low`. Leaderboard ranks use only shared completed prompt IDs and retain each model’s coverage beside its score; expensive models may therefore submit partial suite coverage. The UI must state that provider charges go to the user's account and that candidates/metrics will be public, then require acknowledgement before publication. Export allowlists keep candidates, verifier reports, model route/alias, token and cost telemetry, and timing; they omit provider keys, authorization headers, raw responses, exact request message bodies, and raw provider errors. The uploader rejects duplicate questions, other suites, mixed model/settings rows, and verifier outcomes that fail replay.

Each `run_id` is a single model run. When forming a leaderboard comparison from community rows, select at most one run per model and score the selected runs on their common completed question IDs; do not merge repeated runs for one model into one score. The community publisher requires the Space's `HF_TOKEN` environment secret and never uses a cached developer CLI token.

The Space callback calls `benchmarks.run.run_community_evaluation(model_key, user_api_key, task_limit=12)` for the default Quick run. The runner processes the first 12 frozen eval-v2.1 tasks in order, stops before a task when three maximum request reservations do not fit the per-run USD 10 cap, uses a temporary credential-free spend ledger, and returns `(sanitized_rows, suite_sha256)`. A full run requires `task_limit=72` and `full_run_cost_acknowledged=True`; before collecting that acknowledgement, the UI should show `estimate_community_evaluation_cost(model_key, 72)`, based on measured input/output tokens in the official Beam eval-v2.1 trajectories and the selected model's pinned LiteLLM rates. Before estimating, it requires exactly 72 distinct canonical task IDs, an eval-v2.1 suite ID and digest on every row, and measured token usage on every call. The returned `reference_suite_sha256` is the canonical eval-v2.1 suite digest. This is a rate-card estimate, not a provider billing cap. Each result row carries the server-generated UUID `run_id`; public publication additionally requires the signed-in HF username as `submitted_by`. The runner removes raw completion text and redacts the visitor key before rows reach the UI callback.

The initial public Beam trajectory export is evaluation-only: it records the existing 24-prompt benchmark and must not be presented as a training corpus. The current Pramaan evaluation release uses the predeclared 72-task `eval-v2.1` suite; the earlier `eval-v2` release is retained only for historical reproducibility. The separate 144-row `train-v1.2` suite is the only current prompt source for future training/development records; it has not yet been evaluated or exported. Visitor community evaluations remain in the separate `community/test/` path, always carry `data_role: community_evaluation` and `training_eligible: false`, and must never be promoted into training. A consented development dataset requires its own explicit opt-in, review, and separately versioned `train` release; that ingestion/publishing workflow is not implemented yet. A visitor can download their own run bundle.

## Pramaan Hub releases

- `kshitijthakkar/pramaan-proof-of-art-eval-v2.1` is the current evaluation dataset with `trajectories`, `compact`, and `summary` configs, plus a `previews/` folder. Every row that declares a role must be `benchmark_evaluation` and set `training_eligible: false`. The older `kshitijthakkar/pramaan-proof-of-art-eval-v2` is historical; do not mix its scores or community rows with v2.1.
- `kshitijthakkar/pramaan-proof-of-art-train-v1` contains historical `train-v1` exports and is superseded. The corrected `train-v1.2` prompt digest is recorded in `benchmarks/README.md`; no model outputs or training-eligible export exist for that version yet.
- Both cards use CC BY 4.0, identify Beam, describe the patterns as craft-inspired, cite the tradition sources, and keep evaluation records in a `test` split and training records in a `train` split. The separate prompt-suite digests define the split policy.

The historical export reconstructs request messages from the frozen prompt, profile, and preceding verifier feedback. Its original records did not retain per-call wall-clock timestamps, so `started_at`, `completed_at`, and `verified_at` are null for this backfill; provider-reported latency, token usage, and whole-task elapsed time remain available. The provider did not return billed cost. The exporter applies the published Baseten rate card observed on the same date as this run to produce a clearly labeled estimate; it is not an account invoice. New live trajectories capture actual provider cost if returned and a dated rate-card estimate when token usage and rates are available.

The Beam estimate uses Baseten's published per-million-token prices ($0.40 input, $0.04 cached input, $1.20 output) from the [Baseten pricing page](https://www.baseten.co/pricing/). Hugging Face says routed Inference Provider requests pass through provider rates without markup, while credits and account-level billing determine the amount charged; see [Hugging Face pricing and billing](https://huggingface.co/docs/inference-providers/pricing). The saved estimate uses provider-reported token/cache counts and the recorded rate-card snapshot; it does not subtract monthly credits or negotiated discounts.

## Create the Beam trajectory export

With the local full result file (kept in ignored `benchmarks/results/`):

```bash
python scripts/export_trajectory_dataset.py benchmarks/results/beam-hf-paired-24.jsonl \
  --output benchmarks/data/beam-hf-trajectories-v1.jsonl --model beam-hf
```

The exporter replays/audits benchmark candidates, reconstructs exact prompts for the original run, and refuses to overwrite an existing output. It does not make model calls.

For the v2 benchmark datasets, use `scripts.push_hub_datasets.export_suite_trajectories` as the approved replay gate. It requires full unique suite coverage and the exact committed suite digest, replays each candidate with `parse_pattern` and `verify_pattern`, and checks acceptance and truncation parity before it writes a trajectory export. The legacy `check_trajectory_dataset.py` is not a v2 gate.

To create chat SFT rows from a separately collected training dataset, label its records `data_role: "training"` and `training_eligible: true`, then run:

```bash
python scripts/export_sft_pairs.py path/to/training-trajectories.jsonl --output path/to/sft-pairs.jsonl
```

This flattener refuses to overwrite output and skips evaluation records. It will fail if the input contains no explicitly eligible training calls, so the current Beam benchmark cannot accidentally become a training set.

## Local bounded shard collection

The `benchmarks.collect_shards append` command stores replay-validated local shards under a stable user-supplied shard ID. Every append requires an explicit role, suite digest (derived from the exact suite file), prompt-generator version and seed, plus user-set record, provider-call, token, estimated/provider cost, and elapsed-time caps. The command refuses over-budget imports, mismatched provenance, duplicate IDs with changed content, and suite/role conflicts. Repeating an append with the same ID and provenance is resumable and idempotent; manifest events and records are append-only. These are post-run import caps: they reject a batch after its actual usage is known, and do not interrupt a provider request in flight.

Use `benchmarks.collect_shards merge-training` to create local trajectory, SFT, repair, and preference JSONL exports. It rejects duplicate prompt fingerprints and any train/evaluation overlap. Evaluation shards remain in the local ledger for audit and are never copied to training outputs. Visitor sessions are not collected.

Every trajectory call has a scalar `reward` computed by `pramaan.reward.reward_for_report`, and each record declares `reward_version: "pramaan-reward-v1"` (trajectory schema 1.1). Missing, malformed, parse-failed, schema-failed, and truncated reports receive 0.0. After schema passes, incomplete hard-check sets receive `0.2 + 0.6 * passed / total`; all hard checks passing receives exactly 1.0. The 0.2 floor is intentional as a shaped signal, not acceptance. Since the number and meaning of checks differ by grammar, partial rewards are grammar-local and should not be compared as a cross-grammar quality score. Jaali reports also retain each check's assurance label; a passing approximate web proxy is not an exact geometric proof.

Example (choose a fresh shard ID and set all five caps for each batch):

```bash
python -m benchmarks.collect_shards append --root .pramaan-exports/local --shard-id train-20261030-a --role training --suite benchmarks/suites/train-v1.jsonl --results benchmarks/results/my-bounded-run.jsonl --prompt-generator-version train-v1 --seed 11 --max-records 12 --max-calls 36 --max-tokens 200000 --max-cost-usd 1.00 --max-seconds 300
python -m benchmarks.collect_shards merge-training --root .pramaan-exports/local --output-dir .pramaan-exports/merged-train-20261030
```


## OpenEnv episode and transition dataset

`kshitijthakkar/pramaan-proof-of-art-openenv` stores append-only replay-checked episodes in a `transitions` config: canonical `eval-v2` and `eval-v2.1` transitions use separate uniquely named shards in the `test` split and `train-v1` transitions use the `train` split. Each task has a reset record followed by one step record per submitted proposal. Step records preserve the observation, candidate action, deterministic checks/counterexamples, `pramaan-reward-v1`, terminated/truncated flags, model/provider provenance, and available per-request token, cost, and latency telemetry. The submitted proposal is retained only as the replayable environment action; credentials and raw provider error bodies are excluded. Rows declare `data_role` and `training_eligible`; evaluation data is never training eligible.

Build the shard without provider calls from an existing result file:

```bash
python -c "from pathlib import Path; from scripts.push_hub_datasets import export_openenv_episodes; export_openenv_episodes(rows=__import__('scripts.push_hub_datasets', fromlist=['read_jsonl']).read_jsonl(Path('benchmarks/results/eval-v2.1-beam-20261030.jsonl')), suite_path=Path('benchmarks/suites/eval-v2.1.jsonl'), output_path=Path('benchmarks/data/openenv-eval-v2.1-beam-20261030.jsonl'), role='benchmark_evaluation')"
python scripts/push_hub_datasets.py --repo openenv --openenv-eval-transitions benchmarks/data/openenv-eval-v2.1-beam-20261030.jsonl
```

The exporter replays every action through `PramaanEnvironment`, checks verifier output and reward parity, refuses provider-truncated actions that cannot be faithfully replayed, and will not overwrite a shard. The Hub publisher validates role metadata and refuses an existing shard path. New visitor and local contributions are stored in the separate `kshitijthakkar/pramaan-community-runs` dataset, never in official eval or train datasets. Older Space builds may have written community rows under an eval dataset config; those legacy rows are not copied into this canonical RL collection.

The visitor gallery reads persisted runs through `load_recent_community_runs(limit=100)`. Each new run appends a small timestamped JSON index plus its sanitized JSONL result file; the reader lists index metadata and downloads only the requested recent runs, never the full append-only community corpus. It returns `{run_id, model_key, submitted_at, rows}` newest first and fails with a generic message if the public Hub read is unavailable.

Community submissions are packed locally by `scripts/pack_submission.py` and replay-audited through `pramaan.community.audit_submission_archive` before publication. The resulting rows carry the frozen suite digest, canonical provider host, unverified model label, `run_mode`, and `training_eligible: false`; provider credentials and raw responses are never copied into published rows. The contribution dataset is separate from both official evaluation and training datasets. Space writes use only `HF_COMMUNITY_WRITE_TOKEN`, a fine-grained token scoped to the community repository.
