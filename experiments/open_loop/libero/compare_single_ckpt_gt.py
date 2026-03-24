import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from experiments.robot.robot_utils import get_action
from janus.models import ActionTokenizer, MultiModalityCausalLM, VLChatProcessor


def sanitize_image(image_array):
    image_array = np.asarray(image_array)
    while image_array.ndim > 3:
        image_array = image_array[0]
    return image_array.astype(np.uint8)


def load_stats(stats_path, preferred_key):
    with open(stats_path, "r") as f:
        stats_data = json.load(f)

    if preferred_key in stats_data:
        key = preferred_key
    elif len(stats_data) == 1:
        key = next(iter(stats_data))
    else:
        raise KeyError(
            f"Could not resolve unnorm key. preferred={preferred_key}, available={list(stats_data.keys())}"
        )

    statistic = {
        "action_mask": np.array(stats_data[key]["action"]["mask"]),
        "action_min": np.array(stats_data[key]["action"]["q01"]),
        "action_max": np.array(stats_data[key]["action"]["q99"]),
        "action_dim": int(len(stats_data[key]["action"]["q01"])),
    }
    if "state" in stats_data[key]:
        statistic["state_mask"] = np.array(stats_data[key]["state"]["mask"])
        statistic["state_min"] = np.array(stats_data[key]["state"]["q01"])
        statistic["state_max"] = np.array(stats_data[key]["state"]["q99"])
    return statistic, key


def resolve_model_dir(source_root):
    root = Path(source_root)
    tfmr = root / "tfmr"
    return tfmr if tfmr.exists() else root


def resolve_stats_path(source_root):
    root = Path(source_root)
    candidates = [
        root / "stats_data.json",
        root.parent / "stats_data.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find stats_data.json near {source_root}")


def resolve_episode_path(path_like):
    path = Path(path_like)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("episode_*.npy"))
        if not candidates:
            raise FileNotFoundError(f"No episode_*.npy files found under {path}")
        return candidates[0]
    raise FileNotFoundError(f"Episode path does not exist: {path_like}")


def model_load(model_path, stats_path, cuda_id, horizon, latent_size, use_latent, preferred_key):
    statistic, resolved_key = load_stats(stats_path, preferred_key)
    cfg = SimpleNamespace(
        cuda=str(cuda_id),
        num_open_loop_steps=horizon,
        latent_size=latent_size,
        use_latent=use_latent,
        use_proprio=False,
    )
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_latent=use_latent,
        flow=True,
        action_dim=statistic["action_dim"],
        action_chunk=horizon,
        fast_and_slow=True,
        fast_image_num=1,
    )
    action_tokenizer = ActionTokenizer(tokenizer)
    return cfg, vl_gpt, vl_chat_processor, action_tokenizer, statistic, resolved_key


def get_step_images(step):
    slow = step.get("image_primary", step.get("agentview_image"))
    fast = step.get("image_wrist", step.get("robot0_eye_in_hand_image"))
    if slow is None or fast is None:
        raise KeyError(f"Episode step does not contain expected image keys: {list(step.keys())}")
    return sanitize_image(slow), sanitize_image(fast)


def build_gt_chunk(episode, start_idx, horizon):
    actions = []
    episode_len = len(episode)
    for offset in range(horizon):
        idx = min(start_idx + offset, episode_len - 1)
        action = np.asarray(episode[idx]["action"], dtype=np.float32)
        actions.append(action)
    return np.stack(actions, axis=0)


def plot_chunk(gt_actions, pred_actions, query_idx, output_dir, compare_dim):
    fig, axes = plt.subplots(compare_dim, 1, figsize=(12, max(4, 2.2 * compare_dim)), sharex=True)
    if compare_dim == 1:
        axes = [axes]
    x = np.arange(gt_actions.shape[0])
    for dim in range(compare_dim):
        ax = axes[dim]
        ax.plot(x, gt_actions[:, dim], marker="o", label="GT")
        ax.plot(x, pred_actions[:, dim], marker="x", label="Pred")
        ax.set_ylabel(f"a{dim}")
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend()
    axes[-1].set_xlabel("Horizon Step (0-based)")
    fig.suptitle(f"Open-loop chunk comparison at query step {query_idx}")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"query_{query_idx:04d}.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--episode-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--use-latent", type=int, default=1)
    parser.add_argument("--episode-limit", type=int, default=-1)
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_root)
    stats_path = resolve_stats_path(args.model_root)
    episode_path = resolve_episode_path(args.episode_path)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cfg, model, processor, action_tokenizer, statistic, unnorm_key = model_load(
        str(model_dir),
        str(stats_path),
        args.cuda,
        args.horizon,
        args.latent_size,
        bool(args.use_latent),
        args.task_suite_name,
    )

    episode = np.load(episode_path, allow_pickle=True)
    if args.episode_limit > 0:
        episode = episode[: args.episode_limit]

    task_description = episode[0]["language_instruction"]
    query_indices = list(range(0, len(episode), args.stride))

    results = []
    for query_idx in query_indices:
        step = episode[query_idx]
        slow_image, fast_image = get_step_images(step)
        pred_actions = np.asarray(
            get_action(
                cfg,
                statistic,
                action_tokenizer,
                processor,
                task_description,
                model,
                [Image.fromarray(fast_image)],
                [Image.fromarray(slow_image)],
            ),
            dtype=np.float32,
        )
        gt_actions = build_gt_chunk(episode, query_idx, args.horizon)
        compare_dim = min(gt_actions.shape[1], pred_actions.shape[1])
        plot_chunk(gt_actions, pred_actions[:, :compare_dim], query_idx, args.output_dir, compare_dim)
        results.append(
            {
                "query_idx": query_idx,
                "gt_actions": gt_actions,
                "pred_actions": pred_actions,
            }
        )

    np.savez(
        Path(args.output_dir) / "open_loop_chunks.npz",
        query_indices=np.array([item["query_idx"] for item in results], dtype=np.int32),
        gt_actions=np.stack([item["gt_actions"] for item in results], axis=0),
        pred_actions=np.stack([item["pred_actions"] for item in results], axis=0),
    )

    metadata = {
        "episode_path": args.episode_path,
        "resolved_episode_path": str(episode_path),
        "task_description": task_description,
        "model_root": args.model_root,
        "model_dir": str(model_dir),
        "stats_path": str(stats_path),
        "unnorm_key": unnorm_key,
        "task_suite_name": args.task_suite_name,
        "horizon": args.horizon,
        "stride": args.stride,
        "num_queries": len(results),
        "gt_action_dim": int(results[0]["gt_actions"].shape[1]) if results else None,
        "pred_action_dim": int(results[0]["pred_actions"].shape[1]) if results else None,
        "compared_action_dim": int(min(results[0]["gt_actions"].shape[1], results[0]["pred_actions"].shape[1])) if results else None,
    }
    with open(Path(args.output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
