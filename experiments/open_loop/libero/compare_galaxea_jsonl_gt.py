import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from experiments.robot.robot_utils import get_action
from janus.models import ActionTokenizer, MultiModalityCausalLM, VLChatProcessor


def load_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_episode_id(item):
    fast_images = item.get("input_image_fast", [])
    if not fast_images:
        raise ValueError("input_image_fast is empty")
    sample_path = fast_images[0]
    match = re.search(r"episode(\d+)|episode_(\d+)", sample_path)
    if not match:
        raise ValueError(f"Could not parse episode id from path: {sample_path}")
    return int(next(group for group in match.groups() if group is not None))


def filter_episode_items(items, episode_id):
    selected = [item for item in items if extract_episode_id(item) == episode_id]
    if not selected:
        raise ValueError(f"No rows found for episode_id={episode_id}")
    return selected


def load_stats(stats_path):
    with open(stats_path, "r") as f:
        stats_data = json.load(f)
    if len(stats_data) != 1:
        raise ValueError(f"Expected one top-level stats key, got: {list(stats_data.keys())}")
    key = next(iter(stats_data))
    statistic = {
        "action_mask": np.array(stats_data[key]["action"]["mask"]),
        "action_min": np.array(stats_data[key]["action"]["q01"]),
        "action_max": np.array(stats_data[key]["action"]["q99"]),
        "action_dim": int(len(stats_data[key]["action"]["q01"])),
    }
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
        root.parent / f"{root.parent.name}_train_statistics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find stats file near {source_root}")


def model_load(model_root, cuda_id, horizon, latent_size, use_latent, fast_image_num):
    model_dir = resolve_model_dir(model_root)
    stats_path = resolve_stats_path(model_root)
    statistic, unnorm_key = load_stats(stats_path)
    cfg = SimpleNamespace(
        cuda=str(cuda_id),
        num_open_loop_steps=horizon,
        latent_size=latent_size,
        use_latent=use_latent,
        use_proprio=False,
    )
    processor: VLChatProcessor = VLChatProcessor.from_pretrained(str(model_dir))
    tokenizer = processor.tokenizer
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_latent=use_latent,
        flow=True,
        action_dim=statistic["action_dim"],
        action_chunk=horizon,
        fast_and_slow=True,
        fast_image_num=fast_image_num,
    )
    # The repo expects bf16 later; keep tokenizer setup unchanged.
    action_tokenizer = ActionTokenizer(tokenizer)
    return cfg, model, processor, action_tokenizer, statistic, unnorm_key, model_dir, stats_path


def load_image_list(paths):
    return [Image.open(path).convert("RGB") for path in paths]


def normalize_chunk_for_plot(gt_actions, pred_actions):
    mins = np.minimum(gt_actions.min(axis=0), pred_actions.min(axis=0))
    maxs = np.maximum(gt_actions.max(axis=0), pred_actions.max(axis=0))
    spans = np.maximum(maxs - mins, 1e-6)
    gt_norm = (gt_actions - mins) / spans
    pred_norm = (pred_actions - mins) / spans
    return gt_norm, pred_norm, mins, maxs


def plot_chunk(gt_actions, pred_actions, query_idx, output_dir):
    compare_dim = gt_actions.shape[1]
    gt_norm, pred_norm, mins, maxs = normalize_chunk_for_plot(gt_actions, pred_actions)
    fig, axes = plt.subplots(compare_dim, 1, figsize=(12, max(5, 2.0 * compare_dim)), sharex=True)
    if compare_dim == 1:
        axes = [axes]
    x = np.arange(gt_actions.shape[0])
    for dim in range(compare_dim):
        ax = axes[dim]
        ax.plot(x, gt_norm[:, dim], marker="o", label="GT")
        ax.plot(x, pred_norm[:, dim], marker="x", label="Pred")
        ax.set_ylabel(f"a{dim}\n[{mins[dim]:.3f}, {maxs[dim]:.3f}]")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if dim == 0:
            ax.legend()
    axes[-1].set_xlabel("Horizon Step (0-based)")
    fig.suptitle(f"Galaxea open-loop chunk at query step {query_idx} (per-dim min-max normalized)")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"query_{query_idx:04d}.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--jsonl-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--use-latent", type=int, default=1)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    items = load_jsonl(args.jsonl_path)
    episode_items = filter_episode_items(items, args.episode_id)
    sample_fast_count = len(episode_items[0]["input_image_fast"])

    cfg, model, processor, action_tokenizer, statistic, unnorm_key, model_dir, stats_path = model_load(
        args.model_root,
        args.cuda,
        args.horizon,
        args.latent_size,
        bool(args.use_latent),
        sample_fast_count,
    )

    query_indices = list(range(0, len(episode_items), args.stride))
    results = []
    total_queries = len(query_indices)
    progress_bar = tqdm(query_indices, desc="Open-loop queries", unit="query")
    for query_count, query_idx in enumerate(progress_bar, start=1):
        item = episode_items[query_idx]
        print(
            f"[{query_count}/{total_queries}] query_step={query_idx} "
            f"horizon={args.horizon} dim={len(item['action'][0])}",
            flush=True,
        )
        slow_images = load_image_list(item["input_image_slow"])
        fast_images = load_image_list(item["input_image_fast"])
        pred_actions = np.asarray(
            get_action(
                cfg,
                statistic,
                action_tokenizer,
                processor,
                item["input_prompt"],
                model,
                fast_images,
                slow_images,
            ),
            dtype=np.float32,
        )
        gt_actions = np.asarray(item["action"], dtype=np.float32)
        progress_bar.set_postfix(query_step=query_idx, saved_png=f"query_{query_idx:04d}.png")
        plot_chunk(gt_actions, pred_actions, query_idx, args.output_dir)
        results.append(
            {
                "query_idx": query_idx,
                "gt_actions": gt_actions,
                "pred_actions": pred_actions,
            }
        )

    np.savez(
        Path(args.output_dir) / "open_loop_chunks.npz",
        query_indices=np.array([r["query_idx"] for r in results], dtype=np.int32),
        gt_actions=np.stack([r["gt_actions"] for r in results], axis=0),
        pred_actions=np.stack([r["pred_actions"] for r in results], axis=0),
    )

    metadata = {
        "jsonl_path": args.jsonl_path,
        "episode_id": args.episode_id,
        "model_root": args.model_root,
        "model_dir": str(model_dir),
        "stats_path": str(stats_path),
        "unnorm_key": unnorm_key,
        "horizon": args.horizon,
        "stride": args.stride,
        "num_queries": len(results),
        "action_dim": int(results[0]["gt_actions"].shape[1]) if results else None,
        "task_description": episode_items[0]["input_prompt"] if episode_items else None,
        "fast_image_num": sample_fast_count,
        "plot_normalization": "per_query_per_dim_min_max_over_gt_and_pred",
    }
    with open(Path(args.output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
