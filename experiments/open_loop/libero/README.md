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

Deprecated path for current Galaxea 14D work:
- `compare_single_ckpt_gt.py` was created while trying to compare a 14D dual-arm checkpoint against 7D LIBERO single-arm GT.
- That path is not the correct evaluation setup for Galaxea dual-arm finetuned checkpoints.
- For Galaxea 14D checkpoints, use the dedicated JSONL-based open-loop script that compares against the matching training-format GT.
