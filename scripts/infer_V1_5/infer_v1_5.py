import argparse
import os
import torch

from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates

from PIL import Image


def load_image(image_path: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return image


def infer(
    model_path: str,
    image_path: str,
    prompt: str,
    model_base: str | None = None,
    temperature: float = 0.2,
    top_p: float | None = None,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> str:
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        device=device,
    )

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if getattr(model.config, "mm_use_im_start_end", False):
        query = image_token_se + "\n" + prompt
    else:
        query = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    # 对话模板选择（v1.5 用 llava_v1）
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    # 读图并预处理
    image = load_image(image_path)
    image_sizes = [image.size]
    images_tensor = process_images([image], image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    input_ids = (
        tokenizer_image_token(
            full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(model.device)
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature and temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return output_text


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA v1.5 inference script")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/jusheng/yijia/LLaVA_HCP/checkpoints/llava-v1.5-7b-sft-stage2/checkpoint-45000",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--image",
        type=str,
        default="/home/jusheng/yijia/LLaVA_HCP/images/llava_logo.png",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="decirblr this image",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    assert os.path.exists(args.model_path), f"model path not found: {args.model_path}"
    assert os.path.exists(args.image), f"image not found: {args.image}"
    text = infer(
        model_path=args.model_path,
        image_path=args.image,
        prompt=args.prompt,
        model_base=args.model_base,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    print(text)


if __name__ == "__main__":
    main()


