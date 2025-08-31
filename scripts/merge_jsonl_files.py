#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def build_image_index(data_dir):
    """构建图片文件名到完整路径的索引"""
    image_index = {}
    duplicate_files = defaultdict(list)
    
    data_path = Path(data_dir)
    print("构建图片文件索引...")
    
    # 遍历所有图片文件
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.webp']
    for ext in image_extensions:
        for img_file in tqdm(data_path.rglob(ext), desc=f"扫描{ext}文件"):
            filename = img_file.name
            relative_path = str(img_file.relative_to(data_path))
            
            if filename in image_index:
                duplicate_files[filename].append(relative_path)
                duplicate_files[filename].append(image_index[filename])
            else:
                image_index[filename] = relative_path
    
    # 报告重复文件
    if duplicate_files:
        print(f"发现 {len(duplicate_files)} 个重复文件名:")
        for filename, paths in duplicate_files.items():
            print(f"  {filename}: {paths}")
    
    return image_index, duplicate_files

def extract_dataset_name(filename):
    """从文件名中提取数据集名称"""
    name = filename.replace('.jsonl', '')
    
    # 处理特殊情况
    if name.startswith('ai2d'):
        return 'ai2d'
    elif name.startswith('chartqa'):
        return 'chartqa'
    elif name.startswith('docvqa'):
        return 'docvqa'
    elif name.startswith('dvqa'):
        return 'dvqa'
    elif name.startswith('geoqa'):
        return 'geoqa'
    elif name.startswith('llava_instruct'):
        return 'llava_instruct'
    elif name.startswith('sharegpt4v_instruct'):
        return 'sharegpt4v_instruct'
    elif name.startswith('sharegpt4v_mix'):
        return 'sharegpt4v_mix'
    elif name.startswith('synthdog'):
        return 'synthdog'
    else:
        # 默认取第一个下划线前的部分
        return name.split('_')[0]

def process_jsonl_files(input_dir, data_dir, output_file):
    """处理所有jsonl文件并整合"""
    input_path = Path(input_dir)
    data_path = Path(data_dir)
    
    # 构建图片索引
    image_index, duplicate_files = build_image_index(data_dir)
    
    all_data = []
    missing_images = set()
    duplicate_image_usage = defaultdict(list)
    
    # 获取所有jsonl文件
    jsonl_files = list(input_path.glob('*.jsonl'))
    print(f"找到 {len(jsonl_files)} 个jsonl文件")
    
    total_entries = 0
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            total_entries += sum(1 for _ in f)
    
    with tqdm(total=total_entries, desc="处理JSON条目") as pbar:
        for jsonl_file in jsonl_files:
            dataset_name = extract_dataset_name(jsonl_file.name)
            print(f"处理文件: {jsonl_file.name} -> 数据集: {dataset_name}")
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    pbar.update(1)
                    try:
                        data = json.loads(line.strip())
                        
                        # 修改image路径
                        if 'image' in data:
                            original_image = data['image']
                            if original_image.startswith('images/'):
                                # 提取文件名
                                image_filename = original_image.split('/')[-1]
                                
                                # 在索引中查找
                                if image_filename in image_index:
                                    relative_path = image_index[image_filename]
                                    path_parts = relative_path.split('/')
                                    
                                    # 确保路径格式为 dataset/dataset/...
                                    if len(path_parts) >= 1:
                                        dataset_dir = path_parts[0]
                                        if len(path_parts) >= 2 and path_parts[0] == path_parts[1]:
                                            # 已经是正确的重复格式
                                            formatted_path = relative_path
                                        else:
                                            # 需要重复数据集名称
                                            if len(path_parts) == 1:
                                                formatted_path = f"{dataset_dir}/{dataset_dir}"
                                            else:
                                                formatted_path = f"{dataset_dir}/{dataset_dir}/{'/'.join(path_parts[1:])}"
                                        
                                        # 验证文件是否存在
                                        full_path = data_path / formatted_path
                                        if full_path.exists():
                                            data['image'] = formatted_path
                                        else:
                                            missing_images.add(f"{image_filename} (格式化后路径不存在: {formatted_path})")
                                            continue
                                    else:
                                        data['image'] = relative_path
                                    
                                    # 检查是否使用了重复文件
                                    if image_filename in duplicate_files:
                                        duplicate_image_usage[image_filename].append(f"{jsonl_file.name}:{line_num}")
                                else:
                                    missing_images.add(f"{image_filename} (来自 {jsonl_file.name}:{line_num})")
                                    continue  # 跳过没有找到图片的条目
                        
                        all_data.append(data)
                        
                    except json.JSONDecodeError as e:
                        print(f"警告: {jsonl_file.name} 第{line_num}行JSON解析错误: {e}")
                        continue
    
    # 报告结果
    print(f"\n处理完成:")
    print(f"- 总共处理了 {len(all_data)} 条有效数据")
    print(f"- 成功匹配图片 {len(all_data)} 个")
    
    if missing_images:
        print(f"- 未找到对应图片的条目 {len(missing_images)} 个:")
        for missing in sorted(missing_images):
            print(f"  {missing}")
    
    if duplicate_image_usage:
        print(f"- 使用了重复文件名的条目 {len(duplicate_image_usage)} 个:")
        for filename, usages in duplicate_image_usage.items():
            print(f"  {filename}: {usages}")
    
    # 保存为单个JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据已保存到: {output_file}")
    return len(all_data), len(missing_images)

if __name__ == "__main__":
    input_directory = "playground/data/InternVL-Chat-V1-2-SFT-Data/opensource"
    data_directory = "playground/data/InternVL-Chat-V1-2-SFT-Data/data"
    output_file = "playground/data/merged_sft_data.json"
    
    valid_data_count, missing_count = process_jsonl_files(input_directory, data_directory, output_file)
    
    print(f"\n最终统计:")
    print(f"- 有效数据条目: {valid_data_count}")
    print(f"- 丢弃的条目: {missing_count}")
    print(f"- 成功率: {valid_data_count/(valid_data_count+missing_count)*100:.2f}%")