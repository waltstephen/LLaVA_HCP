# rl_utils.py   (Transformers ≥4.37.2, DataParallel-safe)
# =============================================================================
from __future__ import annotations
from typing import List, Dict, Any, Union

import copy
import torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForImageTextRetrieval,
    AutoModel,
    AutoTokenizer,
)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from torchvision import transforms
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


# -----------------------------------------------------------------------------#
# Helper: unwrap DDP / DataParallel
# -----------------------------------------------------------------------------#
def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


# -----------------------------------------------------------------------------#
# 工具：把 (-mean)/std 归一化 Tensor 还原到 [0,1]
# -----------------------------------------------------------------------------#
def _denorm_tensor(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # Use original dtype for consistency, only convert to float32 for computation if needed
    original_dtype = img.dtype
    img_cpu = img.cpu()
    if img_cpu.dtype != torch.float32:
        img_cpu = img_cpu.float()
    result = (img_cpu * std + mean).clamp(0, 1)
    return result


def _prepare_images(
    images: Union[torch.Tensor, List[torch.Tensor], List[Image.Image]],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> List:
    if isinstance(images, torch.Tensor):
        if images.dim() == 3:  # C,H,W -> 1,C,H,W
            images = images.unsqueeze(0)
        return list(_denorm_tensor(images, mean, std))

    assert isinstance(images, list), "`images` must be Tensor or list"
    first = images[0]
    if isinstance(first, torch.Tensor):
        return [_denorm_tensor(t, mean, std) for t in images]
    return images


# -----------------------------------------------------------------------------#
# 1. 奖励模型
# -----------------------------------------------------------------------------#
class RewardModelCLIP:
    def __init__(self,
                 model_name: str = "openai/clip-vit-large-patch14-336",
                 fp16: bool = True,
                 chunk_size: int = 64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).eval().to(self.device)
        if fp16 and self.device.type == "cuda":
            self.model = self.model.half()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.chunk = chunk_size                       # 分块推理

    # -------- 取分数 --------
    @torch.no_grad()
    def score(self, images, texts: List[str]) -> torch.Tensor:
        B = len(texts)
        assert B == (images.shape[0] if isinstance(images, torch.Tensor) else len(images))

        scores = []
        for i in range(0, B, self.chunk):
            j = min(i + self.chunk, B)
            imgs_chunk = self._prep_pixel_values(images[i:j])    # [b,3,224,224]
            txt_inputs  = self.processor(text=texts[i:j],
                                         return_tensors="pt",
                                         padding=True,
                                         truncation=True).to(self.device)

            img_feat = self.model.get_image_features(pixel_values=imgs_chunk)
            txt_feat = self.model.get_text_features(**txt_inputs)
            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)
            sims = (img_feat * txt_feat).sum(-1)          # [-1,1]
            scores.append(((sims + 1) / 2).cpu())         # → [0,1]

        return torch.cat(scores, 0)

    # -------- 输入适配 --------
    def _prep_pixel_values(self, imgs):
        """
        输出符合 CLIP 要求的 pixel_values：
        shape [b,3,224,224] & 已经归一化到 [-1,1]
        """
        if isinstance(imgs, torch.Tensor):
            # 已经是 vision_tower 的 pixel_values (≈ [-1,1])
            if imgs.min() < 0:
                return imgs.to(self.device).to(self.model.dtype)

            # raw 0~1 / 0~255 Tensor -> 交给 processor 处理
            pv = self.processor(images=imgs, return_tensors="pt")["pixel_values"]
            return pv.to(self.device).to(self.model.dtype)

        # PIL list
        pv = self.processor(images=imgs, return_tensors="pt")["pixel_values"]
        return pv.to(self.device).to(self.model.dtype)


# --------------------  RewardModelBLIP_ITR  --------------------
class RewardModelBLIP_ITR:
    """
    输出 BLIP Image-Text Matching 概率 ∈ [0,1]。
    """
    def __init__(self,
                 model_name="Salesforce/blip-itm-base-coco",
                 fp16=True,
                 chunk_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name).eval().to(self.device)
        if fp16 and self.device.type == "cuda":
            self.model = self.model.half()
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.chunk = chunk_size

    @torch.no_grad()
    def score(self, images, texts: List[str]) -> torch.Tensor:
        B = len(texts)
        assert B == (images.shape[0] if isinstance(images, torch.Tensor) else len(images))

        scores = []
        for i in range(0, B, self.chunk):
            j = min(i + self.chunk, B)
            imgs_chunk = self._prep_pixel_values(images[i:j])
            proc = self.processor(images=None,               # 只用文本分词
                                  text=texts[i:j],
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True).to(self.device)
            out = self.model(pixel_values=imgs_chunk,
                             input_ids=proc["input_ids"],
                             attention_mask=proc["attention_mask"]).itm_score
            prob = out.softmax(-1)[:, 1] if out.dim() == 2 else out.sigmoid()
            scores.append(prob.cpu())

        return torch.cat(scores, 0)

    def _prep_pixel_values(self, images):
        """
        images : List[PIL] | Tensor[B,3,H,W] | List[Tensor]
        returns: Tensor[B,3,224,224]  (with model's dtype, device=self.device)
        """
        # 1) 统一成 List[PIL] 交给 processor
        if torch.is_tensor(images):
            # images: Tensor[B,3,H,W], 可能是 bf16
            # Convert to float32 only for PIL conversion, then back to model dtype
            images_float32 = images.to(dtype=torch.float32)
            images = [transforms.ToPILImage()(img.cpu()) for img in images_float32]
        elif isinstance(images[0], torch.Tensor):
            images = [transforms.ToPILImage()(img.to(dtype=torch.float32).cpu())
                      for img in images]

        # 2) 调用 processor
        pv = self.processor(images=images, return_tensors="pt")["pixel_values"]
        return pv.to(self.device, dtype=self.model.dtype)  # Use model's dtype for consistency


# --------------------  RewardModelDistinctiveness  --------------------
class RewardModelDistinctiveness:
    """
    reward = 1 − max_pairwise_sim，同组回复越不同奖励越高。
    """
    def __init__(self,
                 group_size: int,
                 model_name="sentence-transformers/all-MiniLM-L6-v2",
                 fp16=True,
                 chunk_size=256):
        self.G = group_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).eval().to(self.device)
        if fp16 and self.device.type == "cuda":
            self.enc = self.enc.half()
        self.chunk = chunk_size

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> torch.Tensor:
        embs = []
        for i in range(0, len(texts), self.chunk):
            toks = self.tok(texts[i:i+self.chunk],
                            padding=True,
                            truncation=True,
                            return_tensors="pt").to(self.device)
            h = self.enc(**toks).last_hidden_state.mean(1)
            embs.append(F.normalize(h, dim=-1))
        return torch.cat(embs, 0)          # [N,d]

    @torch.no_grad()
    def score(self, images, texts: List[str]) -> torch.Tensor:
        emb = self._encode(texts)          # [N,d]
        B = len(texts) // self.G
        emb = emb.view(B, self.G, -1)      # [B,G,d]

        sim = torch.einsum("bgd,bkd->bgk", emb, emb)   # [B,G,G]
        sim = sim - torch.eye(self.G, device=sim.device)*2
        max_sim = sim.max(-1).values                   # [B,G]
        return (1 - max_sim).flatten().cpu()           # [0,1]


# -----------------------------------------------------------------------------#
# 2. Generation utilities
# -----------------------------------------------------------------------------#
@torch.no_grad()
def generate_group_responses(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    num_return_sequences: int,
    generation_kwargs: Dict[str, Any] | None = None,
) -> torch.Tensor:
    base_model = _unwrap(model)
    generation_kwargs = {
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 1.0,
        "max_new_tokens": 64,
        "eos_token_id": base_model.config.eos_token_id,
        **(generation_kwargs or {}),
    }

    input_ids = inputs["input_ids"].repeat_interleave(num_return_sequences, dim=0)
    attn = inputs.get("attention_mask", None)
    if attn is not None:
        attn = attn.repeat_interleave(num_return_sequences, dim=0)

    extra = {}
    if "images" in inputs and inputs["images"] is not None:
        imgs = inputs["images"]
        imgs = (
            imgs.repeat_interleave(num_return_sequences, dim=0)
            if isinstance(imgs, torch.Tensor)
            else [img for img in imgs for _ in range(num_return_sequences)]
        )
        extra["images"] = imgs

    dtype = next(base_model.parameters()).dtype
    use_amp = dtype in (torch.bfloat16, torch.float16)
    was_training = base_model.training
    if was_training:
        base_model.eval()

    class CleanLogitsProcessor(LogitsProcessor):
        def __init__(self, clamp_value: float = 50.0) -> None:
            self.clamp_value = float(clamp_value)

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            original_scores = scores
            scores = scores.to(dtype=torch.float32)
            scores = torch.clamp(scores, min=-self.clamp_value, max=self.clamp_value)
            finite_mask = torch.isfinite(scores)
            scores = torch.where(finite_mask, scores, torch.full_like(scores, -float("inf")))
            row_all_masked = torch.isneginf(scores).all(dim=-1)
            if row_all_masked.any():
                idx = original_scores[row_all_masked].argmax(dim=-1)
                scores[row_all_masked] = -float("inf")
                scores[row_all_masked, idx] = 0.0
            return scores.to(dtype=original_scores.dtype)

    processors = LogitsProcessorList([CleanLogitsProcessor()])
    generation_kwargs["logits_processor"] = processors

    with torch.cuda.amp.autocast(enabled=False):
        outputs = base_model.generate(
            input_ids,
            **({"attention_mask": attn} if (attn is not None and "images" not in extra) else {}),
            **extra,
            **generation_kwargs,
        )

    if was_training:
        model.train()

    new_tokens = outputs[:, 1:]  # 从 <bos> 之后全部都是生成的
    B = inputs["input_ids"].size(0)
    new_tokens = new_tokens.view(B, num_return_sequences, -1)
    return new_tokens


# -----------------------------------------------------------------------------#
# 3. 计算 log-probs
# -----------------------------------------------------------------------------#
def _build_forward_inputs(
    input_ids:      torch.LongTensor,      # [B,S]
    generations:    torch.LongTensor,      # [B,G,L]
    G:              int,
    attention_mask: torch.LongTensor | None,
    images: list | torch.Tensor | None,
    device,
):
    B, L = generations.shape[0], generations.shape[-1]
    S    = input_ids.size(1)

    prompt = (
        input_ids.to(device)
        .unsqueeze(1).expand(-1, G, -1)     # [B,G,S]
        .reshape(B * G, S)                  # [B·G,S]
    )
    gen = generations.to(device).reshape(B * G, L)          # [B·G,L]
    ids  = torch.cat([prompt, gen], dim=-1)                  # [B·G,S+L]

    if attention_mask is not None:
        attn_prompt = (
            attention_mask.to(device)
            .unsqueeze(1).expand(-1, G, -1)
            .reshape(B * G, S)
        )
    else:
        attn_prompt = torch.ones_like(prompt, device=device)
    attn = torch.cat([attn_prompt, torch.ones_like(gen)], dim=-1)

    model_inputs = {"input_ids": ids, "attention_mask": attn, "use_cache": False}
    if images is not None:
        if isinstance(images, torch.Tensor):
            images = images.to(device).repeat_interleave(G, 0)
        else:
            images = [img for img in images for _ in range(G)]
        model_inputs["images"] = images
    return model_inputs, L


@torch.no_grad()
def compute_group_log_probs(
    model: torch.nn.Module,
    generations: torch.LongTensor,          # [B, G, L]
    input_ids: torch.LongTensor,            # [B, S]
    attention_mask: torch.LongTensor | None = None,
    images: list | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    句级 log π，返回 [B,G]
    """
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    use_amp = dtype in (torch.float16, torch.bfloat16)
    B, G, _ = generations.shape

    model_inputs, L = _build_forward_inputs(
        input_ids, generations, G, attention_mask, images, device
    )

    with torch.cuda.amp.autocast(dtype=dtype, enabled=use_amp):
        logits = model(**model_inputs).logits[:, :-1]        # [B·G,S+L-1,V]

    labels = model_inputs["input_ids"][:, 1:].clone()        # align
    ignore_mask = labels.eq(IMAGE_TOKEN_INDEX)
    labels[ignore_mask] = IGNORE_INDEX
    gather_idx = labels.masked_fill(labels.eq(IGNORE_INDEX), 0)

    token_logp = (
        logits.log_softmax(-1)
        .gather(2, gather_idx.unsqueeze(-1))
        .squeeze(-1)
    )
    token_logp[ignore_mask] = 0.0

    sent_logp = token_logp[:, -L:].sum(-1).view(B, G)
    return sent_logp


def compute_group_token_log_probs(
    model: torch.nn.Module,
    generations: torch.LongTensor,          # [B,G,L]
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor | None = None,
    images: list | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    token-level log π，返回 [B,G,L] （视觉哨兵位置已置 0）
    """
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    use_amp = dtype in (torch.float16, torch.bfloat16)
    B, G, _ = generations.shape

    model_inputs, L = _build_forward_inputs(
        input_ids, generations, G, attention_mask, images, device
    )

    with torch.cuda.amp.autocast(dtype=dtype, enabled=use_amp):
        logits = model(**model_inputs).logits[:, :-1]

    labels = model_inputs["input_ids"][:, 1:].clone()
    ignore_mask = labels.eq(IMAGE_TOKEN_INDEX)
    labels[ignore_mask] = IGNORE_INDEX
    gather_idx = labels.masked_fill(labels.eq(IGNORE_INDEX), 0)

    token_logp = (
        logits.log_softmax(-1)
        .gather(2, gather_idx.unsqueeze(-1))
        .squeeze(-1)
    )
    token_logp[ignore_mask] = 0.0
    token_logp = token_logp[:, -L:].view(B, G, L)            # 只留生成段
    return token_logp


# -----------------------------------------------------------------------------#
# 4. Stable-GRPO + KL
# -----------------------------------------------------------------------------#
def grpo_stable_loss(
    token_logp_new:  torch.Tensor,   # [B,G,L]  log π_new(a|s)
    rewards:         torch.Tensor,   # [B,G]    缩放到[0,1]
    token_logp_ref:  torch.Tensor | None = None,   # [B,G,L]  log π_ref
    *,
    mode:     str   = "soft",        # "soft" | "adv"
    tau:      float = 0.15,
    clip_w:   float = 5.0,
    kl_beta:  float = 0.02,
) -> torch.Tensor:
    """
    DeepSeek-VL 风格的稳定 GRPO + KL 正则。

    返回：
        总 loss (scalar)
        policy_loss
        kl_loss
    """
    assert mode in ("soft", "adv")

    # -------- 1. 权重 / advantage --------
    if mode == "soft":
        w = torch.softmax(rewards / tau, dim=-1)          # [B,G]
    else:
        mu  = rewards.mean(-1, keepdim=True)
        std = rewards.std( -1, keepdim=True) + 1e-6
        w = (rewards - mu) / std

    w = w.detach().clamp_(-clip_w, clip_w)                # stop-grad, clip
    policy_loss = -(w.unsqueeze(-1) * token_logp_new).mean()

    # -------- 2. KL 正则 (sample-based) --------
    if token_logp_ref is not None and kl_beta > 0:
        kl_token = (token_logp_new - token_logp_ref).detach()  # stop-grad on diff
        kl_loss  = kl_beta * kl_token.mean()
    else:
        kl_loss = torch.zeros(1, device=token_logp_new.device, dtype=token_logp_new.dtype)

    total_loss = policy_loss + kl_loss
    return total_loss, policy_loss.detach(), kl_loss.detach()
