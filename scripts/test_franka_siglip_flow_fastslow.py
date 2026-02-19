import os, sys, pathlib
import argparse
import tqdm
import shutil
import torch
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer

from janus.models import MultiModalityCausalLM, VLChatProcessor, ActionTokenizer
import numpy as np
import os
import PIL.Image
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torchvision
import json
import argparse
import copy
import random
from typing import List, Dict

from termcolor import cprint, colored

import logging
import time
from datetime import datetime

import numpy as np
import pickle

import torch
from dataclasses import dataclass
from PIL import Image

from scipy.spatial.transform import Rotation as R

@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)

def model_load(args):
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        diff=False, flow=True, action_dim=7, fast_and_slow=True, fast_image_num = 2
    )
    action_tokenizer = ActionTokenizer(tokenizer, need_to_sub=3)

    statistics_path = os.path.join(os.path.dirname(args.model_path), "stats_data.json")
    with open(statistics_path, 'r') as f:
        stats_data = json.load(f)
    dataset_name=args.dataset_name

    statistic= {}
    statistic['action_mask'] = np.array(stats_data[dataset_name]['action']['mask'])
    statistic['action_min'] = np.array(stats_data[dataset_name]['action']['q01'])
    statistic['action_max'] = np.array(stats_data[dataset_name]['action']['q99'])
    if args.use_robot_state:
        statistic['state_mask'] = np.array(stats_data[dataset_name]['state']['mask'])
        statistic['state_min'] = np.array(stats_data[dataset_name]['state']['q01'])
        statistic['state_max'] = np.array(stats_data[dataset_name]['state']['q99'])

    return vl_gpt, vl_chat_processor, action_tokenizer, statistic

def model_predict_slow_1_fast_2(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, fast_image, fast_image_wrist, slow_image, state, pointcloud):
    device = f'cuda:{args.cuda}'
    vl_gpt = vl_gpt.to(device).eval()
    parallel_size = 1
    fast_img_len = 2
    slow_img_len = 1
    num_latent_tokens = args.latent_size

    state_tokens = ""
    if args.use_robot_state:
        state = np.array(state, dtype=np.float32)
        normalized_state = np.where(
            statistic['state_mask'],
            np.clip(2 * (state - statistic['state_min']) / (statistic['state_max'] - statistic['state_min'] + 1e-8) - 1, -1, 1),
            state
        )
        state_tokens += action_tokenizer(normalized_state)

    pre_data = []
    input_img_tokens = vl_chat_processor.image_start_tag + vl_chat_processor.image_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag
    
    input_slow_img_tokens = input_img_tokens * slow_img_len
    input_fast_img_tokens = input_img_tokens * fast_img_len

    latent_start_str = "<|latent_start|>"
    latent_pad_str = "<|latent_pad|>" * num_latent_tokens
    latent_end_str = "<|latent_end|>"
    latent_str = latent_start_str + latent_pad_str + latent_end_str
    
    user_content = input_slow_img_tokens + task_description + state_tokens + latent_str + input_fast_img_tokens

    conversation = [
                    {"role": "<|User|>","content": user_content},
                ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
    
    all_image = slow_image + fast_image + fast_image_wrist # all lists, add directly

    with torch.inference_mode():
        input_image_pixel_values = vl_chat_processor.image_processor(all_image, return_tensors="pt")['pixel_values'].to(torch.bfloat16)

        input_ids =  torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))
        tokens = torch.zeros((parallel_size, len(input_ids)), dtype=torch.long)

        for i in range(parallel_size):
            tokens[i, :] = input_ids
            pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=input_image_pixel_values, input_ids=tokens[i], num_image_tokens=[vl_chat_processor.num_image_tokens] * (slow_img_len + fast_img_len)))
        prepare_inputs = vl_chat_processor.batchify(pre_data)

        inputs_embeds = vl_gpt.prepare_inputs_embeds(
            input_ids=tokens.to(device),
            pixel_values=prepare_inputs['pixel_values'].to(torch.bfloat16).to(device),
            images_emb_mask=prepare_inputs['images_emb_mask'].to(device),
            images_seq_mask=prepare_inputs['images_seq_mask'].to(device)
        )
        
        torch.set_printoptions(profile="full")

        input_ids = input_ids.unsqueeze(0)
        latent_indices = (input_ids == 100847).nonzero()
        latent_lists = [[idx[1].item() for idx in latent_indices if idx[0] == i] for i in range(input_ids.shape[0])]
        kv_cache_cot = None
        next_compute_range = (0, latent_indices[:, 1].min().item())

        # inference for latent cot embeddings
        for latent_i in range(num_latent_tokens):
            curr_inputs_embeds = inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :]
            outputs = vl_gpt.language_model.model(
                inputs_embeds=curr_inputs_embeds,
                latent_indexes=torch.arange(0, curr_inputs_embeds.shape[1]).to(device),
                action_indexes=torch.arange(0, 0).to(device),
                use_latent=args.use_latent,
                use_cache=True,
                past_key_values=kv_cache_cot if latent_i!=0 else None # for kv cache
            )
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if latent_i + 1 >= num_latent_tokens
                    else next_compute_range[1] + 1
                ),
            )
            hidden_states = outputs[0][:, -1:, :]
            assert hidden_states.shape[1] == 1
            kv_cache_cot = outputs.past_key_values
            filling_indices = [
                (instance_idx, mask_list[latent_i])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > latent_i
            ]
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair  
                tensor_list[batch_idx][token_idx] = hidden_states[batch_idx][0]
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        noise = torch.randn(inputs_embeds.shape[0], args.action_chunk, 7, device=device)
        samples = vl_gpt.forward_flow(inputs_embeds, noise)
        
        normalized_actions = samples[0].cpu().numpy()
        if normalized_actions.ndim == 1:
            dim = len(normalized_actions)
            if dim == 7 or dim == 14:
                normalized_actions[6] = 0 if normalized_actions[6] < 0.5 else 1
            if dim == 14:
                normalized_actions[13] = 0 if normalized_actions[13] < 0.5 else 1
        else:
            dim = normalized_actions.shape[1]
            if dim == 7 or dim == 14:
                normalized_actions[:, 6] = (normalized_actions[:, 6] >= 0.5).astype(int)
            if dim == 14:
                normalized_actions[:, 13] = (normalized_actions[:, 13] >= 0.5).astype(int)
        actions = np.where(
            statistic['action_mask'],
            0.5 * (normalized_actions + 1) * (statistic['action_max'] - statistic['action_min']) + statistic['action_min'],
            normalized_actions,
        )
        return actions
    

def main(args):
    vl_gpt, vl_chat_processor, action_tokenizer, statistic = model_load(args)

    data = np.load(args.data_path, allow_pickle=True)
    frame_0 = data[0]
    print(frame_0['image_third'])
    fast_image = frame_0['image_third']
    fast_image = [Image.fromarray(fast_image)]
    fast_image_wrist = frame_0['image_wrist']
    fast_image_wrist = [Image.fromarray(fast_image_wrist)]
    slow_image = frame_0['image_third']
    slow_image = [Image.fromarray(slow_image)]

    task_description = frame_0['language_instruction']
    cur_robot_state = frame_0['state']
    point_cloud = frame_0['point_cloud']

    actions = model_predict_slow_1_fast_2(args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, task_description, fast_image, fast_image_wrist, slow_image, cur_robot_state, point_cloud)
    print(actions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--dataset-name', type=str, default='rlbench')
    parser.add_argument('--data-path', type=str, default='')
    parser.add_argument('--use_robot_state', type=int, default=1)
    parser.add_argument('--cuda', type=str, default='7')
    parser.add_argument('--action-chunk', type=int, default=1)
    parser.add_argument('--use_latent', type=int, default=1)
    parser.add_argument('--latent_size', type=int, default=4)
    parser.add_argument('--fs_ratio', type=int, default=4)
    parser.add_argument('--compress_strategy',type=str, required=True,default='average')
    main(parser.parse_args())

    