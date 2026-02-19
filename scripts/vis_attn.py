import os
import argparse
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor, ActionTokenizer
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass
import math

# ==========================================
# 1. 你提供的核心可视化函数 (稍作动态适配)
# ==========================================
def feature_to_score(feat: torch.Tensor, method: str = 'l2', clip_percentile: float = 2.0, target_size=(384, 384)) -> torch.Tensor:
    """
    feat: [B, N_tokens, D]
    Returns: [B, 1, H, W]
    """
    feat = feat.to(torch.float32)

    if method == 'l2':
        score = feat.norm(dim=-1)
    elif method == 'mean':
        score = feat.mean(dim=-1)
    elif method == 'channel_0':
        score = feat[:, :, 0]
    else:
        raise ValueError("Invalid method")

    # [自动计算 Grid Size]
    n_tokens = feat.shape[1]
    grid_size = int(math.sqrt(n_tokens))
    
    # 你的情况应该是 576 -> 24
    if grid_size * grid_size != n_tokens:
        print(f"Warning: Non-square token count {n_tokens}. Grid size set to {grid_size}.")
        
    score = score.view(-1, 1, grid_size, grid_size)
    
    # [修正 1] 使用 target_size (384, 384) 而不是写死 224
    score = torch.nn.functional.interpolate(score, size=target_size, mode='bilinear', align_corners=False)

    # ------------------ 去极值 ------------------
    flat = score.flatten(1)
    lower = torch.quantile(flat, 1 / 100.0, dim=1, keepdim=True)
    upper = torch.quantile(flat, 1 - clip_percentile / 100.0, dim=1, keepdim=True)
    score_clipped = torch.clamp(score, lower.view(-1, 1, 1, 1), upper.view(-1, 1, 1, 1))
    
    # 2. 归一化 0~1
    score_normed = (score_clipped - lower.view(-1, 1, 1, 1)) / (upper - lower + 1e-6).view(-1, 1, 1, 1)
    
    # ================= 修改开始 =================
    
    # 3. 【关键】背景抑制策略
    
    # 步骤 A: 提高 Gamma 指数
    # 原来是 ** 0.5 (会放大噪声)，现在改成 ** 2.0 (抑制噪声)
    score_normed = score_normed ** 2.0
    
    # 步骤 B: 动态阈值截断 (Dynamic Thresholding)
    # 算出当前图中排在前 30% 的分数值，低于这个值的认为是背景噪声
    # 这里的 0.7 表示我们只保留最强的 30% 区域，你可以改大改小 (e.g. 0.6 或 0.8)
    bg_threshold = torch.quantile(score_normed.flatten(1), 0.8, dim=1, keepdim=True)
    score_normed[score_normed < bg_threshold.view(-1, 1, 1, 1)] = 0.0
    
    # 步骤 C: 再次归一化 (可选，为了让剩下的最亮处重新变红)
    score_normed = (score_normed - score_normed.min()) / (score_normed.max() - score_normed.min() + 1e-8)

    # ================= 修改结束 =================

    return score_normed

def gaussian_blur_mask(mask: torch.Tensor, kernel_size=15, sigma=5):
    """
    使用 OpenCV 高斯模糊对 2D torch mask 进行处理
    """
    mask_np = mask.cpu().numpy().astype(np.float32)
    blurred = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), sigma)
    return torch.tensor(blurred, device=mask.device)

def gaussian_blur_mask(mask: torch.Tensor, kernel_size=31, sigma=10):
    """
    使用 OpenCV 高斯模糊对 2D torch mask 进行处理，kernel_size 和 sigma 调大一点以获得更平滑的过渡
    """
    mask_np = mask.cpu().numpy().astype(np.float32)
    # 确保 kernel_size 是奇数
    if kernel_size % 2 == 0: kernel_size += 1
    blurred = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), sigma)
    return torch.tensor(blurred, device=mask.device)

def save_heatmap_on_image_center_masked(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    save_path: str,
    alpha: float = 0.5,
    mask_ratio: float = 0.7,      
    apply_mask: bool = True,      
    mask_decay_ratio: float = 0.2, 
    threshold: float = 0.2        
):
    if image.dim() == 4:
        image = image[0]
    if heatmap.dim() == 4:
        heatmap = heatmap[0]

    image_np = TF.to_pil_image(image.cpu()).convert("RGB")
    image_np = np.array(image_np)
    
    heatmap_np = heatmap.squeeze().cpu().numpy()
    H, W = heatmap_np.shape

    # --- Step 1: Apply Center Mask ---
    if apply_mask:
        mask = torch.full((H, W), mask_decay_ratio, dtype=torch.float32)
        center_h = int(H * mask_ratio)
        center_w = int(W * mask_ratio)
        start_h = (H - center_h) // 2
        start_w = (W - center_w) // 2
        mask[start_h : start_h + center_h, start_w : start_w + center_w] = 1.0
        mask = gaussian_blur_mask(mask, kernel_size=51, sigma=20) 
        mask_np = mask.numpy()
        heatmap_np = heatmap_np * mask_np

    # --- Step 2: Normalize ---
    heatmap_min = np.min(heatmap_np)
    heatmap_max = np.max(heatmap_np)
    if heatmap_max - heatmap_min < 1e-8:
        heatmap_norm = np.zeros_like(heatmap_np)
    else:
        heatmap_norm = (heatmap_np - heatmap_min) / (heatmap_max - heatmap_min)

    # --- Step 3: Hard Thresholding ---
    heatmap_norm[heatmap_norm < threshold] = 0.0

    # --- Step 4: Render & Overlay ---
    cmap = plt.get_cmap('jet')
    heatmap_color = cmap(heatmap_norm)[:, :, :3]
    heatmap_color_uint8 = (heatmap_color * 255).astype(np.uint8)

    image_float = image_np.astype(np.float32)
    heatmap_color_float = heatmap_color_uint8.astype(np.float32)

    overlay_float = image_float * (1.0 - alpha) + heatmap_color_float * alpha
    overlay = np.clip(overlay_float, 0, 255).astype(np.uint8)

    # --- Save ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(overlay)
    ax.axis('off')

    # Colorbar (Fixed)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap='jet'),
        ax=ax, fraction=0.046, pad=0.04
    )
    # [关键修复] 确保 ticks 和 labels 数量一致 (都是3个)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Cold', 'Medium', 'Hot'])
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close()

@dataclass
class VLChatProcessorOutput:
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor
    def __len__(self): return len(self.input_ids)

class ActivationCapturer:
    def __init__(self):
        self.activations = {}

    def get_hook(self, name):
        def hook(model, input, output):
            # LLM Layer output is usually (hidden_states,) tuple
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook

def model_load(args):
    print(f"Loading model from {args.model_path}...")
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        diff=False, flow=True, action_dim=7, fast_and_slow=True
    )
    
    statistics_path = os.path.join(os.path.dirname(args.model_path), "stats_data.json")
    if not os.path.exists(statistics_path):
        statistics_path = os.path.join(os.path.dirname(os.path.dirname(args.model_path)), "stats_data.json")
    
    with open(statistics_path, 'r') as f:
        stats_data = json.load(f)
    dataset_name = list(stats_data.keys())[0]
    statistic = {
        'state_mask': np.array(stats_data[dataset_name]['state']['mask']),
        'state_min': np.array(stats_data[dataset_name]['state']['q01']),
        'state_max': np.array(stats_data[dataset_name]['state']['q99'])
    }
    
    action_tokenizer = ActionTokenizer(tokenizer, need_to_sub=3)
    vl_gpt = vl_gpt.to(f"cuda:{args.cuda}").eval()
    
    return vl_gpt, vl_chat_processor, action_tokenizer, statistic

def process_frame(args, vl_gpt, processor, action_tokenizer, statistic, 
                  capturer, image_np, instruction, state, frame_idx):
    
    device = f'cuda:{args.cuda}'
    target_res = (384, 384) # [Fix] Target resolution for visualization

    fast_image_pil = Image.fromarray(image_np).convert("RGB")
    all_image = [fast_image_pil, fast_image_pil] # Slow + Fast
    
    # State Norm
    state = np.array(state, dtype=np.float32)
    normalized_state = np.where(
        statistic['state_mask'],
        np.clip(2 * (state - statistic['state_min']) / (statistic['state_max'] - statistic['state_min'] + 1e-8) - 1, -1, 1),
        state
    )
    state_tokens = action_tokenizer(normalized_state)

    # Prompt
    img_tokens = processor.image_start_tag + processor.image_tag * processor.num_image_tokens + processor.image_end_tag
    latent_str = "<|latent_start|>" + "<|latent_pad|>" * args.latent_size + "<|latent_end|>"
    
    # Strict Alignment: Slow -> Text -> State -> Latent -> Fast
    user_content = img_tokens + instruction + state_tokens + latent_str + img_tokens
    
    conversation = [{"role": "<|User|>", "content": user_content}]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation, sft_format=processor.sft_format, system_prompt=""
    )

    # Inputs
    with torch.inference_mode():
        pixel_values = processor.image_processor(all_image, return_tensors="pt")['pixel_values'].to(torch.bfloat16).to(device)
        if pixel_values.ndim == 4: pixel_values = pixel_values.unsqueeze(0)
        
        input_ids_cpu = torch.LongTensor(processor.tokenizer.encode(sft_format)).unsqueeze(0)
        
        pre_data = [VLChatProcessorOutput(
            sft_format=sft_format, pixel_values=pixel_values[0], input_ids=input_ids_cpu[0], 
            num_image_tokens=[processor.num_image_tokens]*2
        )]
        prepared = processor.batchify(pre_data)
        
        input_ids = input_ids_cpu.to(device)
        
        inputs_embeds = vl_gpt.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_emb_mask=prepared['images_emb_mask'].to(device),
            images_seq_mask=prepared['images_seq_mask'].to(device)
        )
        
        # Latent Loop
        latent_pad_id = 100847
        latent_indices = (input_ids == latent_pad_id).nonzero()
        latent_lists = [[idx[1].item() for idx in latent_indices if idx[0] == i] for i in range(input_ids.shape[0])]
        
        kv_cache_cot = None
        next_compute_range = (0, latent_indices[:, 1].min().item())
        
        for latent_i in range(args.latent_size):
            curr_inputs_embeds = inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :]
            outputs = vl_gpt.language_model.model(
                inputs_embeds=curr_inputs_embeds,
                latent_indexes=torch.arange(0, curr_inputs_embeds.shape[1]).to(device),
                action_indexes=torch.arange(0, 0).to(device),
                use_latent=args.use_latent,
                use_cache=True,
                past_key_values=kv_cache_cot if latent_i != 0 else None
            )
            next_compute_range = (
                next_compute_range[1],
                (input_ids.shape[1] if latent_i + 1 >= args.latent_size else next_compute_range[1] + 1)
            )
            hidden_states = outputs[0][:, -1:, :]
            kv_cache_cot = outputs.past_key_values
            
            filling_indices = [
                (instance_idx, mask_list[latent_i])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > latent_i
            ]
            for batch_idx, token_idx in filling_indices:
                inputs_embeds[batch_idx, token_idx, :] = hidden_states[batch_idx, 0, :]

        full_latent_indexes = torch.arange(0, inputs_embeds.shape[1]-3-578).to(device)
        full_action_indexes = torch.arange(inputs_embeds.shape[1]-3-578, inputs_embeds.shape[1]).to(device)

        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds,
            latent_indexes=full_latent_indexes,
            action_indexes=full_action_indexes,
            use_latent=args.use_latent,
            use_cache=False,
            output_hidden_states=True 
        )

    # ---------------- Post Process & Visualize ----------------
    num_img_tokens = processor.num_image_tokens
    img_mask = prepared['images_seq_mask'][0].bool()
    img_indices = img_mask.nonzero().squeeze()
    
    # Locate Slow and Fast Image indices
    # We expect 2 blocks of image tokens
    if len(img_indices) >= 2 * num_img_tokens:
        # First block = Slow Image
        slow_start = img_indices[0].item()
        slow_end = img_indices[num_img_tokens-1].item() + 1
        
        # Second block = Fast Image
        fast_start = img_indices[num_img_tokens].item()
        fast_end = img_indices[-1].item() + 1
    else:
        print(f"Warning: Not enough image tokens found ({len(img_indices)}). Skipping frame.")
        return

    # Prepare background image
    base_img_tensor = TF.to_tensor(fast_image_pil.resize(target_res))
    if base_img_tensor.dim() == 3: base_img_tensor = base_img_tensor.unsqueeze(0)

    # Define targets to visualize
    targets = [
        {"name": "slow", "start": slow_start, "end": slow_end},
        {"name": "fast", "start": fast_start, "end": fast_end}
    ]

    for target in targets:
        t_name = target["name"]
        t_start = target["start"]
        t_end = target["end"]
        
        # 1. Vision Encoder Feature
        vision_feat = inputs_embeds[:, t_start:t_end, :] 
        
        # 2. LLM Mid Feature (Layer -16)
        llm_mid_feat = outputs.hidden_states[-16][:, t_start:t_end, :]
        
        # 3. LLM Final Feature (Layer -2)
        llm_final_feat = outputs.hidden_states[-2][:, t_start:t_end, :]
        
        stages = {
            "vision_encoder": vision_feat,
            "llm_mid": llm_mid_feat,
            "llm_final": llm_final_feat
        }
        
        for stage_name, feat in stages.items():
            if feat.shape[1] == 0: continue
            
            try:
                # Feature to Score
                heatmap = feature_to_score(feat, method='l2', clip_percentile=2.0, target_size=target_res)
                
                # Save: frame_0001_slow_vision_encoder.png
                save_filename = f"frame_{frame_idx:04d}_{t_name}_{stage_name}.png"
                save_path = os.path.join(args.output_dir, save_filename)
                
                save_heatmap_on_image_center_masked(
                    image=base_img_tensor, 
                    heatmap=heatmap, 
                    save_path=save_path,
                    alpha=0.5, 
                    apply_mask=True 
                )
            except Exception as e:
                print(f"Error plotting {t_name} {stage_name}: {e}")

def main(args):
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    
    vl_gpt, processor, action_tokenizer, statistic = model_load(args)
    capturer = ActivationCapturer()
    
    print(f"Loading data from {args.data_path}...")
    data = np.load(args.data_path, allow_pickle=True)
    
    for i in tqdm(range(len(data))):
        print(f"Processing frame {i}...")
        input("Press Enter to continue...")

        item = data[i]
        if 'front_image' in item: img = item['front_image']
        elif 'image' in item: img = item['image']
        else: continue
            
        instr = item.get('language_instruction', "")
        # instr = "close the lid on the box"
        state = item.get('state', np.zeros(7))
        
        try:
            process_frame(args, vl_gpt, processor, action_tokenizer, statistic, 
                          capturer, img, instr, state, i)
        except Exception as e:
            print(f"Error frame {i}: {e}")
            import traceback
            traceback.print_exc()
            
        if args.max_frames and i >= args.max_frames: break
    
    print(f"Done! Results saved in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default="vis_results_act")
    parser.add_argument('--cuda', type=str, default="0")
    parser.add_argument('--latent-size', type=int, default=12)
    parser.add_argument('--use-latent', type=int, default=1)
    parser.add_argument('--max-frames', type=int, default=20)
    args = parser.parse_args()
    main(args)