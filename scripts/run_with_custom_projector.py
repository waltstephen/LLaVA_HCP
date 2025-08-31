import argparse
import os
from typing import Dict

import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.multimodal_encoder.builder import build_vision_tower


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


def _resolve_mm_projector_submodule(model):
    # Try common locations for mm_projector
    try:
        sub = model.get_model().mm_projector
        return sub
    except Exception:
        pass
    try:
        sub = model.mm_projector
        return sub
    except Exception:
        pass
    try:
        sub = model.model.mm_projector  # some wrappers
        return sub
    except Exception:
        pass
    # As a last resort, scan named_modules
    for name, module in model.named_modules():
        if name.endswith("mm_projector"):
            return module
    raise AttributeError("Could not locate mm_projector submodule on model")


def load_projector_weights(model, projector_ckpt_path: str) -> None:
    if not os.path.isfile(projector_ckpt_path):
        raise FileNotFoundError(f"Projector checkpoint not found: {projector_ckpt_path}")

    projector_state_dict: Dict[str, torch.Tensor] = torch.load(
        projector_ckpt_path, map_location="cpu"
    )

    # Try to normalize keys to match mm_projector's expected submodule keys
    if any(k.startswith("model.mm_projector.") for k in projector_state_dict.keys()):
        normalized_state_dict = {
            k.replace("model.mm_projector.", ""): v
            for k, v in projector_state_dict.items()
            if k.startswith("model.mm_projector.")
        }
    elif any(k.startswith("mm_projector.") for k in projector_state_dict.keys()):
        normalized_state_dict = {
            k.replace("mm_projector.", ""): v
            for k, v in projector_state_dict.items()
            if k.startswith("mm_projector.")
        }
    else:
        # Assume the state dict directly corresponds to the submodule
        normalized_state_dict = projector_state_dict

    mm_projector = _resolve_mm_projector_submodule(model)
    missing_keys, unexpected_keys = mm_projector.load_state_dict(normalized_state_dict, strict=False)

    if len(unexpected_keys) > 0:
        print(f"[Warning] Unexpected keys in projector checkpoint ignored: {unexpected_keys}")
    if len(missing_keys) > 0:
        print(f"[Warning] Missing keys when loading projector checkpoint: {missing_keys}")


def run_inference(
    base_model_path: str,
    projector_ckpt_path: str,
    image_path: str,
    prompt: str,
    vision_tower_path: str | None = None,
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
):
    model_name = get_model_name_from_path(base_model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        base_model_path,
        model_base=None,
        model_name=model_name,
        device_map=device,
        device=device,
    )

    # Replace projector
    load_projector_weights(model, projector_ckpt_path)

    # Force bind the specified CLIP vision tower if provided
    if vision_tower_path is not None and len(str(vision_tower_path)) > 0:
        try:
            core_model = model.get_model()
        except Exception:
            core_model = model
        # Update config and rebuild vision tower
        setattr(core_model.config, "mm_vision_tower", vision_tower_path)
        vt = build_vision_tower(core_model.config)
        vt.load_model()
        # Align device and dtype with language model
        model_dtype = next(model.parameters()).dtype
        vt.to(device=device, dtype=model_dtype)
        # Attach to model and refresh image_processor
        try:
            core_model.vision_tower = vt
        except Exception:
            # Some wrappers keep it on model
            setattr(model, "vision_tower", vt)
        image_processor = vt.image_processor

    model.to(device)
    model.eval()

    # Load and preprocess image
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_tensor_list = process_images([image], image_processor, model.config)
    # Prefer vision tower device/dtype to avoid device mismatch when model is sharded
    try:
        vision_tower = model.get_vision_tower()
        vt_device = getattr(vision_tower, 'device', device)
        vt_dtype = getattr(vision_tower, 'dtype', next(model.parameters()).dtype)
    except Exception:
        vt_device = device
        vt_dtype = next(model.parameters()).dtype

    image_tensor = image_tensor_list[0].unsqueeze(0).to(dtype=vt_dtype, device=vt_device)

    # Build input with image tokens
    full_prompt = (
        f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n" + prompt
    )
    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids.to(device),
            images=image_tensor,
            do_sample=(temperature > 0),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    # Robust decode: slice by input length; if empty, fallback to full output
    input_token_len = input_ids.shape[1]
    output_seq = output_ids[0]
    if output_seq.dim() == 0:
        output_seq = output_seq.unsqueeze(0)
    sliced_ids = output_seq[input_token_len:]
    generated_ids = sliced_ids if sliced_ids.numel() > 0 else output_seq

    # Debug: raw token ids
    try:
        print("DEBUG: Raw generated token IDs:", generated_ids.tolist())
    except Exception:
        pass

    # Remove any special tokens manually as extra safety
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    gen_list = generated_ids.tolist()
    filtered_ids_list = [tid for tid in gen_list if tid not in special_ids]
    filtered_ids = torch.tensor(filtered_ids_list, dtype=torch.long, device=generated_ids.device)

    generated_text = tokenizer.decode(filtered_ids, skip_special_tokens=True).strip()
    if len(generated_text) == 0:
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("=== Model Output ===")
    print(generated_text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLaVA with a custom projector checkpoint on an image."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/home/jusheng/yijia/LLaVA_HCP/checkpoints/llava-v1.5-7b",
        help="Path to the base LLaVA model directory.",
    )
    parser.add_argument(
        "--projector_ckpt_path",
        type=str,
        default="/home/jusheng/yijia/LLaVA_HCP/checkpoints/llava_1_5_projecter/mm_projector.bin",
        help="Path to the trained projector checkpoint (.bin).",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="/home/jusheng/yijia/LLaVA_HCP/examples/image0.jpeg",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please Describle this image",
        help="User text prompt to ask the model about the image.",
    )
    parser.add_argument(
        "--vision_tower_path",
        type=str,
        default="/home/jusheng/yijia/LLaVA_HCP/checkpoints/clip-vit-large-patch14-336",
        help="Path to CLIP vision tower (e.g., clip-vit-large-patch14-336).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device to run inference on (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        base_model_path=args.base_model_path,
        projector_ckpt_path=args.projector_ckpt_path,
        image_path=args.image_path,
        prompt=args.prompt,
        vision_tower_path=args.vision_tower_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


