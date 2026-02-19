import json

input_file = '/media/liuzhuoyang/LCoT_VLA_MOT/training_data/agilex_mobile_json/arrange_dishes_dual_view1+3_chunk4_fast4_sparse_fastslow_train.json'
output_file = '/media/liuzhuoyang/LCoT_VLA_MOT/scripts/gt.json'

print("正在读取原始JSON文件...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"原始文件包含 {len(data)} 条记录")

# 提取数组索引6032到13529（对应第6033到13530个元素，从1开始计数）
start_idx = 6388  # 第6033个元素（0-based索引是6032）
end_idx = 6435  # 第13530个元素（0-based索引是13529）

# 确保索引在有效范围内
if end_idx >= len(data):
    end_idx = len(data) - 1
    print(f"警告：结束索引超出范围，调整为 {end_idx}")

extracted_data = data[start_idx:end_idx+1]

print(f"提取了 {len(extracted_data)} 条记录（数组索引 {start_idx} 到 {end_idx}）")

# 保存为新文件
print(f"正在保存到 {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_data, f, indent=2, ensure_ascii=False)

print(f"完成！文件已保存到 {output_file}")