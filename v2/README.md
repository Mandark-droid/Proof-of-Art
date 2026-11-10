---
title: Pramaan — AI drew it. Math checked it.
emoji: 🧵
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 8000
hf_oauth: true
pinned: false
license: mit
tags:
  - benchmark
  - openenv
  - reinforcement-learning
  - verifiable-rewards
  - generative-art
  - india
short_description: AI drew it. Math checked it. Proof you can recompute.
suggested_hardware: cpu-basic
---

# Pramaan — AI drew it. Math checked it.

Every tile on this site is something a model actually drew for a brief from an Indian craft tradition, and every one was checked by the same deterministic geometric checker. Pick any tile to see the run behind it: each try, the exact pixels the checker objected to, the reward it earned and the proof you can recompute.

- **Replays**: every model on every brief, try by try.
- **Leaderboard**: the same 72 frozen briefs, three tries each, with cost and failure types.
- **Build with it**: the checker is the benchmark and the reward, packaged as an OpenEnv environment.

Results come from the public [results dataset](https://huggingface.co/datasets/kshitijthakkar/pramaan-proof-of-art-eval-v2.1). A pass means the declared rules hold on a 64 × 64 raster. It says nothing about beauty and never certifies cultural authenticity; the designs are mathematical reinterpretations inspired by craft traditions, not a substitute for the people who keep them alive.

Source: [github.com/Mandark-droid/Proof-of-Art](https://github.com/Mandark-droid/Proof-of-Art).
