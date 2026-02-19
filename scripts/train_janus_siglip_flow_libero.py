import os
import json
import torch
import logging
import argparse
import random
import shutil
import math
import wandb
import PIL.Image
import numpy as np
import time
from safetensors.torch import load_file
from typing import List, Dict, Any
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from einops import rearrange
from transformers import (
    set_seed,
)
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor, ActionTokenizer


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

from dataclasses import dataclass
@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)

def get_custom_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    min_lr_ratio=0.0, 
    num_cycles=0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
        return scaled_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def get_learning_rate(step, initial_lr, num_warmup_steps, num_training_steps, min_lr_ratio, num_cycles=0.5):
    if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps)) * initial_lr
    progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
    scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
    return scaled_factor * initial_lr

def create_component_indexes(seq_len, action_len=7):
    latent_indexes = torch.arange(0, seq_len - action_len)
    action_indexes = torch.arange(seq_len - action_len, seq_len)
    return latent_indexes, action_indexes

class TrainingMetrics:
    def __init__(self, device):
        self.n_step = 0
        self.action_loss = torch.Tensor([0]).to(device=device)
        self.sim_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, action_loss, sim_loss):
        return self.update(action_loss, sim_loss)

    def update(self, action_loss, sim_loss):
        self.n_step += 1
        with torch.no_grad():
            self.action_loss += action_loss.item()
            self.sim_loss += sim_loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.action_loss, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.sim_loss, op=torch.distributed.ReduceOp.SUM)

        action_loss = self.action_loss.item() / (self.world_size * self.n_step)
        sim_loss = self.sim_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.action_loss.fill_(0)
            self.sim_loss.fill_(0)
        return action_loss, sim_loss


class SftDataset(Dataset):
    def __init__(self, config, processor,accelerator, model):
        self.model = model
        self.config = config
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.action_tokenizer = ActionTokenizer(self.tokenizer, need_to_sub=3) # 3 for latent spetial tokens
        self.accelerator = accelerator
        self.image_len = 576
        with open(config.data_path,'r') as f:
            self.data = json.load(f)

        statistics_path = config.data_path.replace(".json", "_statistics.json")
        with open(statistics_path, 'r') as f:
            self.stats_data = json.load(f)

        self.dataset_name = next(iter(self.stats_data))
        self.action_mask = torch.tensor(
            self.stats_data[self.dataset_name]['action']['mask'], 
            dtype=torch.bool
        )
        self.action_min = torch.tensor(
            self.stats_data[self.dataset_name]['action']['q01'], 
            dtype=torch.bfloat16
        )
        self.action_max = torch.tensor(
            self.stats_data[self.dataset_name]['action']['q99'], 
            dtype=torch.bfloat16
        )
        self.state_mask = torch.tensor(
            self.stats_data[self.dataset_name]['state']['mask'], 
            dtype=torch.bool
        )
        self.state_min = torch.tensor(
            self.stats_data[self.dataset_name]['state']['q01'], 
            dtype=torch.bfloat16
        )
        self.state_max = torch.tensor(
            self.stats_data[self.dataset_name]['state']['q99'], 
            dtype=torch.bfloat16
        )
        self.img_dir = os.path.dirname(config.data_path)
        accelerator.print(f'Total data amount: {len(self.data)}')

  
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.data[index]

    def process_image(self,image_paths):
        images = [PIL.Image.open(image_path).convert("RGB") for image_path in image_paths]
        images_outputs = self.processor.image_processor(images, return_tensors="pt")
        return images_outputs['pixel_values']

    def sample_beta(self, alpha, beta, bsize, device):
        alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
        beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
        dist = torch.distributions.Beta(alpha_t, beta_t)
        samples = dist.sample((bsize,))
        return samples.to(dtype=torch.bfloat16)

    def sample_time(self, bsize, device):
        time_beta = self.sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.bfloat16, device=device)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.bfloat16,
            device=device,
        )

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        if self.config.use_latent:
            # Image
            latent_images_nested = [
                [os.path.join(self.img_dir, img) for img in x['output_image'][:8]]
                for x in batch
            ]
            latent_pixel_values_list = [
                self.process_image(img_list).to(torch.bfloat16)  # -> [n_images, C, H, W]
                for img_list in latent_images_nested
            ]
            latent_pixel_values = torch.stack(latent_pixel_values_list, dim=0)

            latent_pointclouds=None
            if self.config.use_pointcloud:
                # Point Cloud
                latent_pc_nested = [
                    [os.path.join(self.img_dir, pc) for pc in x['output_pointcloud']]
                    for x in batch
                ]
                latent_pc_list = []
                for sample_pc_paths in latent_pc_nested:
                    sample_pcs = []
                    for pc_path in sample_pc_paths:
                        pc_data = np.load(pc_path)
                        sample_pcs.append(torch.from_numpy(pc_data).float())
                    latent_pc_list.append(torch.stack(sample_pcs)) # [4, N_points, C]
                latent_pointclouds = torch.stack(latent_pc_list).to(torch.bfloat16) # [B, 4, N_points, C]
            
            latent_states=None
            if self.config.use_latent_robot_state:
                # State
                states = [x['output_state'] for x in batch]
                states = torch.tensor(states, dtype=torch.bfloat16).reshape(len(states), -1, self.config.action_dim)
                latent_states = torch.where(
                    self.action_mask.to(states.device),
                    torch.clamp(2 * (states - self.state_mask.to(states.device)) / (self.state_max.to(states.device) - self.state_min.to(states.device) + 1e-8) - 1, -1, 1),
                    states
                )
        else:
            latent_pixel_values = None
            latent_pointclouds = None
            latent_states = None

        input_img_tokens = self.processor.image_start_tag + self.processor.image_tag*self.processor.num_image_tokens +self.processor.image_end_tag

        # Generate noisy actions and timesteps for diffusion
        actions = [x['action'] for x in batch]
        actions = torch.tensor(actions, dtype=torch.bfloat16).reshape(len(actions), -1, self.config.action_dim)

        normalized_actions = torch.where(
            self.action_mask.to(actions.device),
            torch.clamp(2 * (actions - self.action_min.to(actions.device)) / (self.action_max.to(actions.device) - self.action_min.to(actions.device) + 1e-8) - 1, -1, 1),
            actions
        )

        time = self.sample_time(normalized_actions.shape[0], normalized_actions.device)
        time_expanded = time[:, None, None]

        noise = self.sample_noise(normalized_actions.shape, normalized_actions.device)

        x_t = (time_expanded * noise + (1 - time_expanded) * normalized_actions)
        u_t = (noise - normalized_actions)

        normalized_state=None
        if self.config.robot_state:
            states = [x['state'] for x in batch]
            states = torch.tensor(states, dtype=torch.bfloat16).reshape(len(states), -1, self.config.action_dim)
            normalized_state = torch.where(
                self.state_mask.to(states.device),
                torch.clamp(2 * (states - self.state_min.to(states.device)) / (self.state_max.to(states.device) - self.state_min.to(states.device) + 1e-8) - 1, -1, 1),
                states
            )

        # Init latent special tokens
        latent_start_str = "<|latent_start|>"
        latent_pad_str = "<|latent_pad|>" * self.config.latent_size
        latent_end_str = "<|latent_end|>"

        if self.config.action_branch_image == 'main':
            for x in batch: x['input_image'] = [x['input_image'][0], x['input_image'][0]]
        elif self.config.action_branch_image == 'wrist':
            for x in batch: x['input_image'] = [x['input_image'][0], x['input_image'][1]]
        elif self.config.action_branch_image == 'none':
            for x in batch: x['input_image'] = [x['input_image'][0]]
        else:
            assert False, f"Invalid action branch image: {self.config.action_branch_image}"

        # Prepare data in batch
        pre_data = []
        for x in batch:
            img_len = len(x['input_image']) if 'input_image' in x and len(x['input_image']) > 0 else 0

            latent_str = latent_start_str + latent_pad_str + latent_end_str if self.config.use_latent else ""
            prompts = input_img_tokens + x['input_prompt'] + latent_str

            if self.config.action_branch_image != 'none':
                prompts += input_img_tokens

            conversation = [
                {"role": "<|User|>","content": prompts},
            ]

            pre_format = self.processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processor.sft_format,
                system_prompt="",
            )
            sft_format = pre_format
            
            if img_len > 0:
                encoder_pixel_values = self.process_image([os.path.join(self.img_dir,input_img) for input_img in x['input_image']])
                num_image_tokens = [self.image_len] * img_len
            else:
                encoder_pixel_values = None
                num_image_tokens = []
            
            input_ids = torch.LongTensor(self.processor.tokenizer.encode(sft_format))
            pre_data.append(
                VLChatProcessorOutput(
                    sft_format=sft_format, 
                    pixel_values=encoder_pixel_values, 
                    input_ids=input_ids, 
                    num_image_tokens=num_image_tokens
                )
            )

        if len(pre_data) > 0:
            prepare_inputs = self.processor.batchify(pre_data)

        return {
            "input_ids": prepare_inputs.input_ids,
            "encoder_pixel_values": prepare_inputs.pixel_values.to(torch.bfloat16),
            "latent_pixel_values": latent_pixel_values,
            "latent_pointclouds": latent_pointclouds, # [B, 4, N, C]
            "latent_states": latent_states, # [B, 4, 7]
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "robot_state": normalized_state,
            "attention_mask": prepare_inputs.attention_mask,
            "images_seq_mask": prepare_inputs['images_seq_mask'],
            "images_emb_mask": prepare_inputs['images_emb_mask'],
        }


def save_checkpoint(
    model,
    processor,
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    global_step: int,
    is_last: bool = False,
    stats_data = None
) -> None:

    save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
    
    if accelerator.is_main_process:
        # Manage checkpoint numbers
        checkpoint_files = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
        if args.max_ckpts > 0 and len(checkpoint_files) >= args.max_ckpts:
            oldest_ckpt = min(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
            shutil.rmtree(os.path.join(args.output_dir, oldest_ckpt))

        os.makedirs(save_dir, exist_ok=True)
        output_dir = os.path.join(save_dir, 'tfmr')

        model.save_pretrained(output_dir, state_dict=accelerator.get_state_dict(model))
        processor.save_pretrained(output_dir)

        with open(os.path.join(save_dir, 'stats_data.json'), 'w') as f:
            json.dump(stats_data, f, indent=2)
            
        logger.info(f"Statistics have been saved to {os.path.join(save_dir, 'stats_data.json')}")

    accelerator.wait_for_everyone()
    logger.info(f'Checkpoint {epoch}-{global_step} saved successfully')



def train(args: argparse.Namespace) -> None:

    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Set random seed
    set_seed(args.seed)

    if accelerator.is_main_process:
        wandb.init(
            project=args.experiment_name,
            name=args.run_name,
            config=args,
            dir=args.log_dir,
        )
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = (
        args.train_bsz_per_gpu * 
        dist.get_world_size() * 
        accelerator.gradient_accumulation_steps
    )

    processor = VLChatProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        flow = True,
        action_dim=args.action_dim,
        use_pointcloud=args.use_pointcloud,
        use_latent=args.use_latent,
        robot_state=args.robot_state,
        local_files_only=True,
        action_branch_image=args.action_branch_image,
        single_branch=args.single_branch,
    )
    model_config = model.config

    if args.use_latent and not args.freeze_latent and args.use_pointcloud: # pointcloud embedder initialization
        model.projector_3d.initialize_weights()
        model.load_encoder_to_pointcloud_embedder(args.pointcloud_embedder_ckpt_path) # load pointcloud embedder

    if args.load_action_from_pretrain:
        model_action_pretrain = load_file(args.action_pretrain_path, device="cpu")

    for name, param in model.named_parameters():
        if '_action' in name:
            if args.load_action_from_pretrain:
                base_name = name.replace('_action', '')
                if base_name in model_action_pretrain.keys():
                    param.data.copy_(model_action_pretrain[base_name])
                    accelerator.print(f"Initialized {name} from action pretrain {base_name}")
                else:
                    assert False, f"Cannot find {base_name} in action pretrain"
            if args.freeze_action:
                param.requires_grad = False
            else:
                param.requires_grad = True
        elif 'x_embedder' in name or 't_embedder' in name or 'final_layer' in name:
            if args.load_action_from_pretrain:
                if name in model_action_pretrain.keys():
                    param.data.copy_(model_action_pretrain[name])
                    accelerator.print(f"Initialized {name} from action pretrain {name}")
                else:
                    assert False, f"Cannot find {name} in action pretrain"
            if args.freeze_action:
                param.requires_grad = False
            else:
                param.requires_grad = True
        elif args.robot_state and 'state_embedder' in name:
            if args.load_action_from_pretrain:
                if name in model_action_pretrain.keys():
                    param.data.copy_(model_action_pretrain[name])
                    accelerator.print(f"Initialized {name} from action pretrain {name}")
            if args.freeze_action:
                param.requires_grad = False
            else:
                param.requires_grad = True
        elif any(name.startswith(prefix) for prefix in ["vision_model", "aligner", "gen_vision_model", "gen_aligner", "gen_embed", "gen_head"]):
            param.requires_grad = False
        elif 'pointcloud_embedder' in name:
            param.requires_grad = False
        elif 'projector_3d' in name:
            param.requires_grad = True
        else:  ## only for a llm (latent_llm, embed_tokens, norm, lm_head)
            if args.freeze_latent:
                param.requires_grad = False
            else:
                param.requires_grad = True

    accelerator.print("\n==== Parameter Freeze Status ====\n")
    for name, param in model.named_parameters():
        status = "TRAINABLE" if param.requires_grad else "FROZEN"
        accelerator.print(f"{name:60}  {status}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    accelerator.print(f"Freeze latent: {args.freeze_latent}")
    accelerator.print(f"Load action from pretrain: {args.load_action_from_pretrain}")
    accelerator.print(f"Total parameters: {total_params/1e9:.2f}B")
    accelerator.print(f"Trainable parameters: {trainable_params/1e9:.2f}B")
    accelerator.print(f"Non-trainable parameters: {non_trainable_params/1e9:.2f}B")
    accelerator.print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = SftDataset(args, processor, accelerator, model)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bsz_per_gpu,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4
    )

    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps // dist.get_world_size()
    lr_scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio
    )
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    metric = TrainingMetrics(device=torch.cuda.current_device())
    model.train()
    global_step = 0

    for epoch in range(0, args.n_epochs):
        train_iter = tqdm(train_dataloader, total=len(train_dataloader)) if accelerator.is_main_process else train_dataloader
        for batch in train_iter:

            inputs_embeds = model.prepare_inputs_embeds(
                    input_ids=batch['input_ids'],
                    pixel_values=batch['encoder_pixel_values'],
                    images_emb_mask=batch['images_emb_mask'],
                    images_seq_mask=batch['images_seq_mask']
                )
            
            # torch.set_printoptions(profile="full")
            # print(batch['input_ids'])
            # print(batch['input_ids'].shape)
            # print(batch['noisy_actions'][0])
            # print("before:", inputs_embeds.shape)
            # input("Press Enter to continue...")

            if args.robot_state:
                robot_state = model.state_embedder(batch['robot_state'])

            noisy_actions = model.x_embedder(batch['noisy_actions'].to(inputs_embeds.dtype))
            timesteps = model.t_embedder(batch['timesteps'].to(inputs_embeds.dtype)).unsqueeze(1)
            inputs_embeds = torch.cat([
                inputs_embeds,
                robot_state if args.robot_state else torch.empty(0, dtype = torch.bfloat16, device=inputs_embeds.device),
                timesteps,
                noisy_actions,
            ], dim=1)
            batch['attention_mask'] = torch.cat([
                batch['attention_mask'],
                torch.ones((batch['attention_mask'].shape[0], robot_state.shape[1]), dtype=torch.bool).to(batch['attention_mask'].device) if args.robot_state else torch.empty((batch['attention_mask'].shape[0], 0), dtype=torch.bool, device=batch['attention_mask'].device),
                torch.ones((batch['attention_mask'].shape[0], timesteps.shape[1]), dtype=torch.bool).to(batch['attention_mask'].device),
                torch.ones((batch['attention_mask'].shape[0], noisy_actions.shape[1]), dtype=torch.bool).to(batch['attention_mask'].device),
            ], dim=1)

            # print("after: ", inputs_embeds.shape)
            # input("after check shape")

            action_expert_tokens_num = 1+1+args.action_chunks if args.robot_state else 1+args.action_chunks
            if args.action_branch_image != 'none':
                action_expert_tokens_num += (args.image_token_num + 2)
            if args.single_branch:
                latent_indexes, action_indexes = create_component_indexes(inputs_embeds.shape[1], inputs_embeds.shape[1])
            else:
                latent_indexes, action_indexes = create_component_indexes(inputs_embeds.shape[1], action_expert_tokens_num)
            
            if args.use_latent:
                bs, n_future = batch['latent_pixel_values'].shape[0:2]

                # Process helper images
                helper_images = rearrange(batch['latent_pixel_values'], "b n c h w -> (b n) c h w")
                img_embeds_flat = model.aligner(model.vision_model(helper_images)) # [B*4, 576, D]
                img_embeds = rearrange(img_embeds_flat, "(b n) t d -> b n t d", b=bs, n=n_future) # Reshape -> [B, 4, 576, D]
                # Compression (Average Pooling)
                img_embeds = img_embeds.view(bs, n_future, args.compressed_imgs_tokens, args.image_token_num // args.compressed_imgs_tokens, -1)
                compressed_imgs = img_embeds.mean(dim=3)  # [B, n_future, 4, D]
                
                compressed_pcs = torch.empty(0, device=compressed_imgs.device, dtype=compressed_imgs.dtype)
                if args.use_pointcloud:
                    # Process helper pointclouds
                    helper_pcs = batch['latent_pointclouds'].to(img_embeds.device).to(img_embeds.dtype) # [B, 4, N_points, C]
                    helper_pcs_flat = rearrange(helper_pcs, "b n p c -> (b n) p c") # Flatten -> [B*4, N_points, C]
                    pc_embeds_flat, pc_centers = model.pointcloud_embedder(helper_pcs_flat)# Encode -> [B*4, T_pc_raw, D]
                    pc_embeds_projected = model.projector_3d(pc_embeds_flat.to(torch.bfloat16) )  # Project -> [B*4, T_pc_raw, D_model]
                    pc_embeds = rearrange(pc_embeds_projected, "(b n) t d -> b n t d", b=bs, n=n_future) # Reshape -> [B, 4, T_pc_raw, D]
                    # Compression (Average Pooling)
                    compressed_pcs = pc_embeds.mean(dim=2, keepdim=True)
                
                compressed_latent_robot_states = torch.empty(0, device=compressed_imgs.device, dtype=compressed_imgs.dtype)
                if args.use_latent_robot_state:
                    # Process helper states
                    latent_robot_states = batch['latent_states'].to(img_embeds.device)
                    state_embeds_full = model.state_embedder(latent_robot_states) # # [B, 4, 7, D]
                    # Compress state embedding (Average Pooling)
                    compressed_latent_robot_states = state_embeds_full.mean(dim=2, keepdim=True) # [B, 4, 1, D]

                combined_embeds = torch.cat([compressed_imgs, compressed_pcs, compressed_latent_robot_states], dim=2)
                compressed_latent_embeds = rearrange(combined_embeds, "b n k d -> b (n k) d")
                compressed_latent_embeds = compressed_latent_embeds.to(inputs_embeds.dtype)

                assert compressed_latent_embeds.shape[1] == args.latent_size

                latent_start_idx = -(action_expert_tokens_num + 1 + args.latent_size + 1)
                inputs_embeds[:, latent_start_idx+1 : latent_start_idx+1+compressed_latent_embeds.shape[1], :] = compressed_latent_embeds

                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch['attention_mask'],
                    return_dict=True,
                    use_cache=False, 
                    latent_indexes=latent_indexes.to(inputs_embeds.device), 
                    action_indexes=action_indexes.to(inputs_embeds.device),
                    use_latent=args.use_latent,
                )
                hidden_states = outputs.last_hidden_state

                inferred_embeddings_all = hidden_states[:,latent_start_idx : latent_start_idx+args.latent_size, :]
                similarity = F.cosine_similarity(inferred_embeddings_all, compressed_latent_embeds, dim=-1).mean()
                sim_loss = 1.0 - similarity
                loss = torch.tensor(0.0).to(sim_loss.device)
                loss += sim_loss

                action_loss = torch.tensor(0.0).to(sim_loss.device)
                if args.latent_action_same_time_train:
                    predicted_noise = model.final_layer(hidden_states)[:, -(batch['target'].shape[1]):, :]
                    action_loss = nn.MSELoss()(predicted_noise, batch['target'].to(predicted_noise.dtype))
                    loss += action_loss
                else:
                    if not args.freeze_action:
                        inputs_embeds[:, latent_start_idx+1 : latent_start_idx+1+args.latent_size, :] = inferred_embeddings_all
                        outputs = model.language_model.model(
                            inputs_embeds=inputs_embeds,
                            attention_mask=batch['attention_mask'],
                            return_dict=True,
                            use_cache=False, 
                            latent_indexes=latent_indexes.to(inputs_embeds.device), 
                            action_indexes=action_indexes.to(inputs_embeds.device),
                            use_latent=args.use_latent,
                        )
                        hidden_states = outputs.last_hidden_state
                        predicted_noise = model.final_layer(hidden_states)[:, -(batch['target'].shape[1]):, :]
                        action_loss = nn.MSELoss()(predicted_noise, batch['target'].to(predicted_noise.dtype))
                        loss = action_loss

                metric(action_loss, sim_loss)

            else:
                latent_indexes=torch.arange(0, 0).to(inputs_embeds.device)
                action_indexes=torch.arange(0, inputs_embeds.shape[1]).to(inputs_embeds.device)
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch['attention_mask'],
                    return_dict=True,
                    use_cache=False,
                    latent_indexes=latent_indexes,
                    action_indexes=action_indexes,
                    use_latent=args.use_latent,
                )
                hidden_states = outputs.last_hidden_state
                
                predicted_noise = model.final_layer(hidden_states[:, -(batch['target'].shape[1]):, :]) # the last token is noise
                action_loss = nn.MSELoss()(predicted_noise, batch['noise'])
                loss = action_loss
                print(f"action_loss: {action_loss:.6f}")
                sim_loss = torch.tensor(0.0).to(action_loss.device)
                metric(action_loss, sim_loss)

            accelerator.backward(loss)
            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                action_loss, sim_loss= metric.get_metric()
                if accelerator.is_main_process:
                    train_iter.set_postfix(
                        epoch=epoch,
                        step=global_step,
                        total_steps=len(train_dataloader),
                        skip=accelerator.optimizer_step_was_skipped,
                        length=len(batch["input_ids"][0]),
                        action_loss=f"{action_loss:.6f}",
                        sim_loss=f"{sim_loss:.6f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                    wandb.log({
                        'action_loss': action_loss,
                        'sim_loss': sim_loss,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }, step=global_step)
            global_step += 1

        if ((epoch + 1) % args.save_freq == 0) or (epoch == args.n_epochs-1):
            accelerator.wait_for_everyone()
            save_checkpoint(
                model=model,
                processor=processor, 
                accelerator=accelerator,
                args=args,
                epoch=epoch,
                step=global_step-1,
                global_step=global_step,
                is_last=(epoch == args.n_epochs-1),
                stats_data=train_dataset.stats_data,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-training parameter configuration')
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, default='janus_train', help='Experiment name')
    parser.add_argument('--run_name', type=str, default='run_1', help='Run name')
    parser.add_argument('--model_path', type=str, default='', help='Pre-trained model path')

    # Data related
    parser.add_argument('--data_path', type=str, required=True, help='Training data path, can be multiple paths')
    parser.add_argument('--data_root', type=str, required=True, default='')
    parser.add_argument('--output_dir', type=str, default='./', help='Model save path')
    parser.add_argument('--max_ckpts', type=int, default=10, help='Maximum number of checkpoints to save')
    parser.add_argument('--log_dir', type=str, default='./train_logs', help='Log save path')

    # Training related
    parser.add_argument('--max_seq_len', type=int, default=4096, help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping threshold, set to 0 for no clipping')
    parser.add_argument('--train_bsz_per_gpu', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--min_lr_ratio', type=float, default=0., help='Minimum learning rate ratio to peak learning rate')
    parser.add_argument('--warmup_rates', type=float, default=0., help='Warmup ratio')
    parser.add_argument('--n_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')

    # Others
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--action_dim', type=int, default=7, help='action dim')
    parser.add_argument('--action_chunks', type=int, default=1, help='action chunks')
    parser.add_argument('--robot_state', type=int, default=0, help='enable robot state')
    parser.add_argument('--load_action_from_pretrain', type=int, default=0)
    parser.add_argument('--freeze_latent', type=int, default=0)
    parser.add_argument('--freeze_action', type=int, default=0)
    parser.add_argument('--image_token_num', type=int, default=576)
    parser.add_argument('--use_latent', type=int, default=0)
    parser.add_argument('--use_pointcloud', type=int, default=0)
    parser.add_argument('--use_latent_robot_state', type=int, default=0)
    parser.add_argument('--latent_size', type=int, default=4)
    parser.add_argument('--pointcloud_embedder_ckpt_path', type=str, default="", help='PointCloud embedder checkpoint path')
    parser.add_argument('--action_pretrain_path', type=str, required=True, help='Action pretrain checkpoint path')

    parser.add_argument('--latent_action_same_time_train', type=int, default=0, help='Latent and action train at the same time')
    parser.add_argument('--action_branch_image', type=str, default='main', help='Action branch image')
    parser.add_argument('--single_branch', type=int, default=0, help='Single branch')
    parser.add_argument('--compressed_imgs_tokens', type=int, default=1, help='Compressed imgs tokens')

    args = parser.parse_args()
    
    # Set paths
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    train(args)     

