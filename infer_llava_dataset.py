#!/usr/bin/env python3
# infer_llava_dataset.py
"""
随机从训练数据集抽 1 条 (image, prompt) 做推理
兼容两种 checkpoint 结构：
1) output_dir/
      ├─ mm_projector.bin
      └─ config.json
2) output_dir/checkpoint-1234/mm_projector.bin
"""

import os, glob, json, random, argparse, re, torch
from PIL import Image
from transformers import AutoTokenizer

from llava.mm_utils import process_images, tokenizer_image_token
from llava.conversation import conv_templates
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

# ----------------------------- CLI ----------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_json",
                    default="./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json")
parser.add_argument("--image_folder",
                    default="./playground/data/LLaVA-Pretrain/images")
parser.add_argument("--checkpoint_root", default="./checkpoints/llava-v1.5-7b-pretrain")
parser.add_argument("--base_model", default="lmsys/vicuna-7b-v1.5")
parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14-336")
parser.add_argument("--index", type=int, default=None, help="固定样本索引")
parser.add_argument("--device", default="cuda", help="cuda / cpu / cuda:0 ...")
args = parser.parse_args()
device = torch.device(args.device)

# ----------------- 辅助：解析数据集 ----------------- #
def load_dataset(path):
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

# ----------------- 辅助：寻找 adapter ---------------- #
def find_adapter(checkpoint_root: str) -> str:
    """
    返回 mm_projector.bin 的绝对路径
    逻辑顺序：
      1) 如果 checkpoint_root 本身就是文件 → 直接返回
      2) root/mm_projector.bin 存在 → 返回
      3) root/checkpoint-*/mm_projector.bin 找最新 → 返回
    """
    # 1) root 本身是文件
    if os.path.isfile(checkpoint_root):
        return checkpoint_root

    # 2) root/mm_projector.bin
    direct = os.path.join(checkpoint_root, "mm_projector.bin")
    if os.path.isfile(direct):
        return direct

    # 3) root/checkpoint-*/mm_projector.bin
    cand = glob.glob(os.path.join(checkpoint_root, "checkpoint-*", "mm_projector.bin"))
    if not cand:
        raise FileNotFoundError(f"在 {checkpoint_root} 未找到 mm_projector.bin")
    cand.sort(key=lambda p: int(os.path.basename(os.path.dirname(p)).split("-")[-1]))
    return cand[-1]

# ----------------- 1. 读数据 & 取样本 ---------------- #
records = load_dataset(args.dataset_json)
idx = args.index if args.index is not None else random.randrange(len(records))
rec = records[idx]

# prompt：取 conversations 中第一条 human
prompt_text = None
if "conversations" in rec:
    for msg in rec["conversations"]:
        if msg.get("from") == "human":
            prompt_text = msg.get("value", "")
            break
if prompt_text is None:
    raise ValueError("record 缺少 human prompt")

prompt_text = re.sub(r"<\s*image\s*>", "", prompt_text, flags=re.IGNORECASE).strip()

img_path = os.path.join(args.image_folder, rec["image"])
if not os.path.exists(img_path):
    raise FileNotFoundError(img_path)

print(f"[INFO] 样本 #{idx}")
print(f"[INFO] prompt: {prompt_text}")
print(f"[INFO] image : {img_path}")

# ----------------- 2. 解析 adapter ------------------ #
adapter_path = find_adapter(args.checkpoint_root)
print(f"[INFO] 使用 adapter: {adapter_path}")

# ----------------- 3. 加载模型 ---------------------- #
tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)

model = LlavaLlamaForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# -------- 关键修正：构造一个 model_args -------- #
from types import SimpleNamespace
vision_args = SimpleNamespace(
    vision_tower            = args.vision_tower,   # openai/clip-vit-large-patch14-336
    mm_vision_select_layer  = -2,                  # 与训练时保持一致
    mm_vision_select_feature= "patch",                # 默认
    pretrain_mm_mlp_adapter = None,                # 这里不加载预训练 adapter
    mm_patch_merge_type     = "flat",              # 或你训练用的其它取值
    mm_projector_type       = "mlp2x_gelu",        # 与训练一致
    mm_use_im_start_end     = False,
)

model.get_model().initialize_vision_modules(vision_args)
# --------------------------------------------------- #

# 再把你训练好的 mm_projector 权重覆盖进去
adapter_state = torch.load(adapter_path, map_location="cpu")
model.load_state_dict(adapter_state, strict=False)
model.get_model().mm_projector.to(dtype=model.dtype, device=device)

model.eval().to(device)

# ----------------- 4. 构造输入（完全对齐训练逻辑） ----------------- #
from PIL import Image
from llava.constants import DEFAULT_IMAGE_TOKEN

# ❶ 读取图片并做预处理 ------------------------------------------------
raw_image = Image.open(img_path).convert("RGB")
vision_tower = model.get_vision_tower()
if isinstance(vision_tower, list):          # FSDP 场景返回 list
    vision_tower = vision_tower[0]
image_processor = vision_tower.image_processor

image_tensor = (
    process_images([raw_image], image_processor, model.config)
    .to(device=device, dtype=model.dtype)
)

# ❷ 重新拼 conversation（严格复刻 preprocess_plain） ---------------
#    • human 的内容必须含 <image>
#    • assistant 位置留空，等待模型生成
conv = conv_templates["plain"].copy()       # 与训练参数 --version plain 对齐
human_msg = rec["conversations"][0]["value"]          # 原始文本，含 <image>
if DEFAULT_IMAGE_TOKEN not in human_msg:
    # 极端情况：数据清洗被改动；手动补回
    human_msg = DEFAULT_IMAGE_TOKEN + "\n" + human_msg

# Plain 模板里 human 只放 <image>；如果你希望完全对齐训练时的行为，
# 可以把下面这一行改成 human_msg = DEFAULT_IMAGE_TOKEN
conv.append_message(conv.roles[0], human_msg)

# assistant 占位
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()                  # 拼成完整 prompt 字符串

# ❸ 将 <image> 占位符替换成真正的特殊 token，并分词 ---------------
# ↓ tokenizer_image_token 返回 1-D
input_ids_1d = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")

# -> 变成 2-D [1, L]
input_ids = input_ids_1d.unsqueeze(0).to(device)

# ----------------- 5. 生成 ------------------------- #
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=128,
        do_sample=False,
    )[0]

answer = tokenizer.decode(output_ids[input_ids.shape[-1]:],
                          skip_special_tokens=True).strip()

print("\n=================  模型回答  =================")
print(answer)
print("=============================================")
