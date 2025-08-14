import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor, ActionTokenizer
from dataclasses import dataclass
import json

@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)

def process_image(image_paths,vl_chat_processor):
    images = [PIL.Image.open(image_path).convert("RGB") for image_path in image_paths]
    images_outputs = vl_chat_processor.image_processor(images, return_tensors="pt")
    return images_outputs['pixel_values']


# Define text+image-to-image generation function
def text_and_image_to_image_generate(input_prompt, state_tokens, input_image_path, output_path, vl_chat_processor, vl_gpt, action_tokenizer, statistic, temperature = 1.1, parallel_size = 2):
    torch.cuda.empty_cache()

    input_img_tokens = vl_chat_processor.image_start_tag + vl_chat_processor.image_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag + vl_chat_processor.image_start_tag + vl_chat_processor.pad_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag

    pre_data = []
    input_images = [input_image_path]
    img_len = len(input_images)
    prompts = input_img_tokens * img_len + input_prompt + state_tokens
    conversation = [
                    {"role": "<|User|>","content": prompts},
                    {"role": "<|Assistant|>", "content": ""}
                ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )

    image_token_num_per_image = 576
    action_token_num = 7
    img_size = 384
    patch_size = 16

    with torch.inference_mode():
        input_image_pixel_values = process_image(input_images,vl_chat_processor).to(torch.bfloat16).cuda()
        quant_input, emb_loss_input, info_input = vl_gpt.gen_vision_model.encode(input_image_pixel_values)
        image_tokens_input = info_input[2].detach().reshape(input_image_pixel_values.shape[0], -1)
        image_embeds_input = vl_gpt.prepare_gen_img_embeds(image_tokens_input)

        input_ids =  torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))
        new_value = torch.tensor(207).expand(*input_ids.shape[:-1], 1)
        input_ids = torch.cat([input_ids, new_value], dim=-1)

        # torch.set_printoptions(threshold=10_000)
        # print(input_ids)

        tokens = torch.zeros((parallel_size, len(input_ids)), dtype=torch.long)

        for i in range(parallel_size):
            tokens[i, :] = input_ids
            pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=input_image_pixel_values, input_ids=tokens[i], num_image_tokens=[vl_chat_processor.num_image_tokens] * img_len))
        prepare_inputs = vl_chat_processor.batchify(pre_data)

        # torch.set_printoptions(threshold=10_000)
        # print(tokens)

        inputs_embeds = vl_gpt.prepare_inputs_embeds(
                    input_ids=tokens.cuda(),
                    pixel_values=prepare_inputs['pixel_values'].to(torch.bfloat16).cuda(),
                    images_emb_mask=prepare_inputs['images_emb_mask'].cuda(),
                    images_seq_mask=prepare_inputs['images_seq_mask'].cuda()
                )
 
        image_gen_indices = (tokens == vl_chat_processor.image_end_id).nonzero()
        for ii, ind in enumerate(image_gen_indices):
            if ii % 2 == 0:
                offset = ind[1] + 2
                inputs_embeds[ind[0],offset: offset+image_embeds_input.shape[1],:] = image_embeds_input[(ii // 2) % img_len]

        ### ------generate action "mode 2" -------- #####
        generate_ids = vl_gpt.language_model.generate(inputs_embeds=inputs_embeds, max_new_tokens=7)
        print(generate_ids)

        ### ------generate action "mode 2" -------- #####
        # generated_action_tokens = torch.zeros((parallel_size, action_token_num), dtype=torch.int).cuda()
        # for i in range(action_token_num):
        #     outputs = vl_gpt.language_model.model(inputs_embeds=cur_inputs_embeds if i != 0 else inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        #     hidden_states = outputs.last_hidden_state

        #     logits = vl_gpt.language_model.lm_head(hidden_states[:, -1, :])

        #     # ch: ------ #
        #     probs = torch.softmax(logits / temperature, dim=-1)
        #     next_token = torch.multinomial(probs, num_samples=1)
        #     # next_token = torch.argmax(logits, dim=-1, keepdim=True)
        #     # ch: ------ #

        #     generated_action_tokens[:, i] = next_token.squeeze(dim=-1)
        #     next_token = next_token.view(-1)
        #     action_emb = vl_gpt.language_model.get_input_embeddings()(next_token)
        #     cur_inputs_embeds = action_emb.unsqueeze(dim=1)
        # print(generated_action_tokens)
        
        normalized_actions = action_tokenizer.decode_token_ids_to_actions(generate_ids.cpu().numpy())

        if normalized_actions.ndim == 1 and len(normalized_actions) == 7:
            normalized_actions[6] = np.where(normalized_actions[6] < 0.5, 0, 1)
        elif normalized_actions.ndim == 1 and len(normalized_actions) == 14:
            normalized_actions[6] = np.where(normalized_actions[6] < 0.5, 0, 1)
            normalized_actions[13] = np.where(normalized_actions[13] < 0.5, 0, 1)
        elif normalized_actions.ndim > 1:
            if normalized_actions.shape[1] == 7:
                normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
            elif normalized_actions.shape[1] == 14:
                normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
                normalized_actions[:, 13] = np.where(normalized_actions[:, 13] < 0.5, 0, 1)

        actions = np.where(
            statistic['action_mask'],
            0.5 * (normalized_actions + 1) * (statistic['action_max'] - statistic['action_min']) + statistic['action_min'],
            normalized_actions,
        )
        print(actions)

        add_tokens = torch.cat([generate_ids, torch.tensor([[100001, 100016]]*generate_ids.shape[0]).cuda()], dim=-1)
        add_embeds = vl_gpt.language_model.get_input_embeddings()(add_tokens)
        inputs_embeds = torch.cat([inputs_embeds, add_embeds], dim=1)


        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
        for i in range(image_token_num_per_image):
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state

            logits = vl_gpt.gen_head(hidden_states[:, -1, :])

            # ch: ------ #
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # next_token = torch.argmax(logits, dim=-1, keepdim=True)
            # ch: ------ #

            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = next_token.view(-1)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        output_images = []
        for i in range(parallel_size):
            save_path = output_path.replace('.png','') + f'_{i}.png'
            PIL.Image.fromarray(visual_img[i]).save(save_path)
            output_images.append(save_path)
        return output_images



# Load model and processor
model_path = "/media/chenhao/ShareGPT-4o-Image/exp/action_image/together_1e-5/checkpoint-49-6200/tfmr"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,torch_dtype=torch.bfloat16
)
vl_gpt = vl_gpt.cuda().eval()
action_tokenizer = ActionTokenizer(tokenizer)


# Run
prompt = "close box"
input_image_path = "/media/chenhao/T2I-R1/rlbench_data/close_box/episode0/image0.png"
image_output_path = "/media/chenhao/ShareGPT-4o-Image/vis/test_output.png"

statistics_path = "/media/chenhao/ShareGPT-4o-Image/training_data/json/train_statistics.json"
with open(statistics_path, 'r') as f:
    stats_data = json.load(f)
dataset_name = next(iter(stats_data))
statistic= {}
statistic['action_mask'] = np.array(stats_data[dataset_name]['action']['mask'])
statistic['action_min'] = np.array(stats_data[dataset_name]['action']['q01'])
statistic['action_max'] = np.array(stats_data[dataset_name]['action']['q99'])
statistic['state_mask'] = np.array(stats_data[dataset_name]['state']['mask'])
statistic['state_min'] = np.array(stats_data[dataset_name]['state']['q01'])
statistic['state_max'] = np.array(stats_data[dataset_name]['state']['q99'])

cur_robot_state = [0.278490275144577, -0.008158989250659943, 1.4719393253326416, -3.141590940498379, 0.24234042527843602, 3.1415862939198944, 1.0]
state = np.array(cur_robot_state, dtype=np.float32)
normalized_state = np.where(
    statistic['state_mask'],
    np.clip(2 * (state - statistic['state_min']) / (statistic['state_max'] - statistic['state_min'] + 1e-8) - 1, -1, 1),
    state
)
state_tokens = ""
state_tokens += action_tokenizer(normalized_state)
text_and_image_to_image_generate(prompt, state_tokens, input_image_path, image_output_path, vl_chat_processor, vl_gpt, action_tokenizer, statistic, parallel_size = 4)