# LIBERO Open-Loop Workspace

This directory is reserved for LIBERO open-loop comparison work.

Current scope:
- Reuse the existing LIBERO evaluation entrypoint to verify that a checkpoint can run.
- Keep all future open-loop scripts and saved comparison artifacts in one place.

Suggested workflow:
1. Use `run_pretrain_rollout.sh` to verify a pretrained checkpoint can run end-to-end.
2. Later add a dedicated open-loop export script here that saves fixed inputs, predicted actions, and GT actions.
3. Run the same export script on both pretrained and finetuned checkpoints, then plot the comparison.

Notes:
- The current wrapper still calls the existing rollout evaluator, not a true open-loop exporter.
- When the open-loop exporter is added, save intermediate outputs under `outputs/`.
