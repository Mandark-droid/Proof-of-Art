# Pramaan benchmark protocol

*AI drew it. Math checked it.* Proof of art that you can verify.

Pramaan's published comparison is the completed `eval-v2.1` run. It evaluates eleven models on the same 72 frozen tasks across 36 Indian states and Union Territories. The suite digest is `b04b568d84d2568fd261789819d4e925844dba5afe9c5962275ee571d84e4e3a`; the public export contains 792 model-task rows.

## Shared evaluation

Every model received the same ordered task IDs, initial prompt, deterministic verifier, 4,096-token output ceiling, and maximum of three proposals per task. Provider retries were disabled. All eleven models completed all 72 tasks. The verifier and result aggregation path were shared. Provider-specific sampling parameters are recorded in each trajectory because the providers do not expose identical controls.

The roster is Beam; Claude Fable 5.1, Fable 5.2, Haiku 5.5, Opus 5.5, and Sonnet 5.5; DeepSeek V4 Flash; GPT-6 Astra and Luna; and GPT-6.1 Sol and Terra. Scores use all 72 task IDs. Per-model pass@1/pass@3, token use, latency, estimated rate-card cost, and spend limits are reported in [the benchmark results](../benchmarks/README.md).

## Recorded results and interpretation

The Beam baseline scored 17/72 pass@1 (23.6%) and 47/72 pass@3 (65.3%). Its trajectories contain 146 candidate replays and 46 unique accepted masks. The other model results are a single completed run per model; they are observed benchmark outcomes, not confidence estimates for future runs. Cost figures are rate-card estimates because provider-reported billed cost was unavailable.

One frozen task, `py_kolam-as-practised-in-puducherry-easy-35`, requests one loop on a 2×2 D2 Kolam grid. Exhaustive enumeration shows the profile permits zero or two loops, never one. It remains in the shared denominator for every model, has no reference pattern, and was not rerun. The reference set therefore contains 71 verified designs for the other tasks.

The geometric checker measures declared shapes and constraints; it does not measure whether the drawing is visibly legible. Of 536 accepted designs, 51 show little or no visible ink. A geometric pass also does not establish cultural authenticity, physical print behavior, or continuous vector properties. Jaali minimum-web thickness is an approximate raster proxy; other assurance labels are retained in the verifier reports.

## Community submissions

Visitor submissions are separate from the official evaluation and training datasets. The submitted model name is self-reported and unverified. A submission is published only after explicit consent, sanitization, and deterministic replay audit; published rows carry `data_role: community_evaluation` and `training_eligible: false`. Community results do not change the official benchmark scores. Compare at most one run per model on common completed task IDs and show coverage. See the [trajectory dataset policy](../benchmarks/TRAJECTORY_DATASET.md).

## Reproducibility and data handling

The frozen suite, protocol version, prompt and suite digests, model route, request settings, verifier version, proposal sequence, feedback, outcomes, and available token, latency, and cost telemetry are recorded in the trajectories. Credentials, authorization headers, and raw provider error bodies are excluded from published records. Raw completions are retained only where the dataset's stated access and privacy policy permits them.

Evaluation records are not training eligible. The corrected 144-brief `train-v1.2` suite is separate from `eval-v2.1`; it has not been evaluated or exported as training records. Historical suites and exports remain available only for reproducibility and are not mixed with the current comparison.

For runner details, exports, and known limitations, see [the benchmark report](../benchmarks/README.md). For trajectory fields and split policy, see [the trajectory dataset specification](../benchmarks/TRAJECTORY_DATASET.md).
