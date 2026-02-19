import torch
import argparse
from transformers import AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    # 添加必要的参数以匹配你的 config，防止加载报错
    parser.add_argument('--action_dim', type=int, default=7)
    parser.add_argument('--use_latent', type=int, default=1)
    args = parser.parse_args()

    model_path = "/media/liuzhuoyang/LCoT_VLA_MOT/exp/latent_cot_mot_flow/janus_pro_siglip_uni3d_1B_1e-4_mot_sparse_slow1_fast2_pc_state_6_1218/stage2/checkpoint-199-5800/tfmr"

    print(f"Loading model from {model_path}...")
    
    # 注意：这里尽量使用 cpu 加载以节省显存，或者加上 device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        diff=False,
        flow=True,
        action_dim=args.action_dim,
        use_pointcloud=False,
        use_latent=args.use_latent,
        ignore_mismatched_sizes=True,
    )

    def count_parameters(module):
        return sum(p.numel() for p in module.parameters())

    print("\n" + "="*40)
    print("       Model Parameter Breakdown       ")
    print("="*40)

    # 1. 打印总体参数
    total = count_parameters(model)
    print(f"Total Model Size: {total / 1e9:.3f} B")

    # 2. 细分模块
    # 根据 Janus 代码库的常见命名习惯
    components = [
        ("Vision Encoder (vision_model)", "vision_model"),
        ("Aligner (aligner)", "aligner"),
        ("LLM (language_model)", "language_model"),
        ("Gen Vision (gen_vision_model)", "gen_vision_model"),
        ("Action Embedder (x_embedder)", "x_embedder"),
        ("Time Embedder (t_embedder)", "t_embedder"),
        ("Final Layer (final_layer)", "final_layer"),
    ]

    for name, attr in components:
        if hasattr(model, attr):
            module = getattr(model, attr)
            params = count_parameters(module)
            if params > 1e9:
                print(f"{name:<30}: {params / 1e9:.3f} B")
            else:
                print(f"{name:<30}: {params / 1e6:.3f} M")
        else:
            print(f"{name:<30}: [Not Found]")

    print("="*40)

if __name__ == "__main__":
    main()