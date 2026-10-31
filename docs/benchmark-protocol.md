# Pramaan benchmark protocol

This note records the planned one-pass comparison protocol. The run must not start until the selected prompt manifest, its ordered task IDs and digest, provider settings, cost preflight, and release revision have been reviewed. The existing partial 72-task provider run is aborted and excluded from scores and public datasets.

## Shared suite and scoring

The source suite is `eval-v2.1` at `benchmarks/suites/eval-v2.1.jsonl` (72 tasks across 36 states and Union Territories). The comparison uses one fixed 47-task subset selected before any model call. Every model receives the same task IDs in the same order, the same initial prompt, the same 4,096-token completion limit, the same deterministic verifier, and up to three proposals per task. Report pass@1, pass@3, repair attempts, verifier outcomes, task coverage, token use, latency, and cost separately for every model. Compare quality only over the exact shared completed task IDs; disclose missing tasks.

Select the 47 tasks using seed `pramaan-eval-v2.1-common47-v1`, independently of model results. Rank state names by UTF-8 SHA-256 of `seed|pair|state` (lexical state name breaks a tie); include both the easy and hard task for the first 11 states. Rank the remaining states by SHA-256 of `seed|single|state`; include the easy task for the first 14 and the hard task for the other 11. Preserve original suite order in the final manifest. The subset therefore contains 25 easy and 22 hard tasks and represents every state. Publish the exact ordered IDs and digest before execution.

## Model arms and settings

Planned arms are Beam (`reflection/Beam-1.0-501B-A23B`), Claude Haiku 5.5, Sonnet 5.5, Opus 5.5, Fable 5.1, and DeepSeek V4.1 Flash through CometAPI. The five paid models each have an independent hard cap of USD 10, for a maximum paid-model spend of USD 50; a cap on one model does not reduce another model's budget. Beam is the baseline. Use one benchmark runner, one suite manifest, one normalizer and one scoring path for all arms. Record the exact endpoint, model ID, request parameters, package versions, prompt and suite digests, every raw response, verifier feedback, errors, retries, per-call input/output tokens, request/provider/verifier durations, and estimated or reported cost.

Use `max_tokens=4096` and up to three proposals per task for every arm. Beam and DeepSeek use temperature 0.4. The Claude API rejects `temperature`, so Claude uses provider-default sampling with `output_config.effort=low`; disclose this provider limitation. SDK and transport retries are disabled. A failure is recorded, not silently retried by a different path. Do not change the token limit, prompt, subset, normalizer, proposal count or scoring after the run starts. The intended comparison is one pass, with no reruns.

## Cost gate

The earlier USD 85 shared Anthropic cap and its preflight estimate are superseded. Before any paid request, regenerate the ordered task manifest and exact digest, calculate a conservative bound for each paid model using current provider rates and prompt sizes, confirm that each independent USD 10 hard cap is enforced and the aggregate ceiling is USD 50, and disclose which arms are expected to reach all 47 tasks. If a model's bound exceeds its cap, that arm may stop at the cap; do not silently reduce its token limit, change its task sequence, or alter the proposal count. Compare quality only over the exact shared completed task IDs and disclose each model's task coverage. The user must review the concrete no-call preflight before calls begin. Provider-reported billed cost is distinct from a locally estimated rate-card cost and must be labeled accurately.

## Output handling and data eligibility

All arms use the same strict output normalizer: remove at most one complete outer JSON code fence and leading `<think>…</think>` blocks only when followed directly by a JSON object. Do not extract JSON from prose or repair malformed output. Preserve raw responses and record every cleanup action. Keep evaluation trajectories in the evaluation split and mark them ineligible for training. Do not add aborted, incomplete, community, or unreviewed runs to benchmark scores or training data. Community contributions require explicit consent, sanitization, and deterministic replay before publication.

## Current baseline and references

The audited single Beam run on all 72 `eval-v2.1` tasks scored 17/72 pass@1 (23.6%) and 47/72 pass@3 (65.3%) over 146 candidate replays, with 46 unique accepted masks. All 72 rows passed independent deterministic replay with no provider errors. Estimated rate-card cost was USD 0.4382; provider-billed cost was unavailable. This is a single stochastic run, not a cross-model ranking, and must not be combined with the aborted partial run.

The frozen v1 suite contains 24 Ajrakh prompts. It is a different task set and should not be directly compared with `eval-v2.1`. The old `eval-v2` scores and `train-v1` prompt/run are superseded. The corrected 144-brief `train-v1.1` suite has not been evaluated and has no matching trajectory or training export yet.

For runner commands, validated exports, legacy summaries, and known limitations, see [the benchmark report](../benchmarks/README.md). For trajectory fields and split policy, see [the trajectory dataset specification](../benchmarks/TRAJECTORY_DATASET.md).
