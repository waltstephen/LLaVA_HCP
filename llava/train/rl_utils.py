# rl_utils.py   (Transformers ≥4.37.2, DataParallel-safe)
# =============================================================================
from __future__ import annotations
from typing import List, Dict, Any, Union, Tuple

import copy
import random
import torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torch.distributed as dist
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForImageTextRetrieval,
    AutoModel,
    AutoTokenizer,
)
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
        返回符合 CLIP 要求的 pixel_values：
        - 如果已经是 vision_tower 的 pixel_values（≈ [-1, 1]），直接返回到 device/dtype；
        - 否则把输入规范到 [0,1] 再交给 self.processor，最后返回到 device/dtype。
        """
        def _is_vision_pixel_values(x: torch.Tensor) -> bool:
            # 经验性判断：已标准化到大约 [-1,1]（min<0 且 max<=1.05）
            # 兼容少量数值抖动
            return x.dtype.is_floating_point and x.min() < 0.0 and x.max() <= 1.05

        def _to_float32(x: torch.Tensor) -> torch.Tensor:
            if x.dtype in (torch.bfloat16, torch.float16):
                return x.to(torch.float32)
            return x if x.dtype == torch.float32 else x.to(torch.float32)

        # --- Tensor 批输入 ---
        if isinstance(imgs, torch.Tensor):
            # case A: 已经是 vision_tower 的 [-1,1] 像素
            if _is_vision_pixel_values(imgs):
                return imgs.to(self.device).to(self.model.dtype)

            # case B: 原始图像张量，需要通过 processor
            x = _to_float32(imgs)
            x_min = float(x.min())
            x_max = float(x.max())

            # 规范到 [0,1]
            if x_max > 1.0:
                if x_min >= 0.0 and x_max <= 255.0:
                    x = torch.clamp(x / 255.0, 0.0, 1.0)
                else:
                    # 值域异常，直接裁剪
                    x = torch.clamp(x, 0.0, 1.0)
            else:
                # 已经在 [0,1]，保持不变；若有轻微越界也裁剪
                if x_min < 0.0 or x_max > 1.0:
                    x = torch.clamp(x, 0.0, 1.0)

            pv = self.processor(images=x, return_tensors="pt")["pixel_values"]
            return pv.to(self.device).to(self.model.dtype)

        # --- 列表输入（PIL / Tensor 混合） ---
        proc_inputs = []
        is_all_vision_pixel_values = True
        for im in imgs:
            if isinstance(im, torch.Tensor):
                if _is_vision_pixel_values(im):
                    # 已是 [-1,1]，直接保留（不进 processor）
                    proc_inputs.append(im)
                else:
                    is_all_vision_pixel_values = False
                    t = _to_float32(im)
                    t_min = float(t.min())
                    t_max = float(t.max())
                    if t_max > 1.0:
                        if t_min >= 0.0 and t_max <= 255.0:
                            t = torch.clamp(t / 255.0, 0.0, 1.0)
                        else:
                            t = torch.clamp(t, 0.0, 1.0)
                    else:
                        if t_min < 0.0 or t_max > 1.0:
                            t = torch.clamp(t, 0.0, 1.0)
                    proc_inputs.append(t)
            else:
                # PIL.Image / numpy 之类，交给 processor 处理
                is_all_vision_pixel_values = False
                proc_inputs.append(im)

        if is_all_vision_pixel_values:
            # 全部都是已经标准化的 pixel_values（罕见但支持）
            batch = torch.stack(proc_inputs, dim=0) if isinstance(proc_inputs[0], torch.Tensor) else proc_inputs
            return batch.to(self.device).to(self.model.dtype)

        # 需要 processor 的路径（支持 PIL / 0~1 Tensor）
        pv = self.processor(images=proc_inputs, return_tensors="pt")["pixel_values"]
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

    import accelerate.utils.operations as accel_ops

    # patch: 防止 generate 内部把 logits 转 float32
    if not hasattr(accel_ops, "_old_convert_to_fp32"):
        accel_ops._old_convert_to_fp32 = accel_ops.convert_to_fp32
        accel_ops.convert_to_fp32 = lambda x: x

    with torch.cuda.amp.autocast(dtype=dtype, enabled=use_amp):
        with torch.no_grad():
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
        kl_token = token_logp_new - token_logp_ref.detach()  # stop-grad on diff
        kl_loss  = kl_beta * kl_token.mean()
    else:
        kl_loss = torch.zeros(1, device=token_logp_new.device, dtype=token_logp_new.dtype)

    total_loss = policy_loss + kl_loss
    return total_loss, policy_loss.detach(), kl_loss.detach()


# -----------------------------------------------------------------------------#
# 5. CaRPO Components: Calibration+Risk, Rank-first, PR-MD with KL control
# -----------------------------------------------------------------------------#

class CalibratedRewardAggregator(nn.Module):
    """
    维护每个 RM 的 EMA μ/σ，做 z-score 校准；
    融合并加上风险惩罚（组内方差）。

    输入: s_bgk ∈ [B,G,K]
    输出: fused ∈ [B,G], rm_var ∈ [B,G], z_bgk ∈ [B,G,K]
    """
    def __init__(
        self,
        K: int,
        alphas: List[float],
        risk_lambda: float = 0.2,
        ema_m: float = 0.95,
        eps: float = 1e-6,
        clamp: float | None = None,
        use_tanh: bool = False,
    ) -> None:
        super().__init__()
        self.K = int(K)
        self.register_buffer("mu", torch.zeros(self.K))
        self.register_buffer("var", torch.ones(self.K))
        self.register_buffer("_prev_mu", torch.zeros(self.K))
        self.register_buffer("_prev_sigma", torch.ones(self.K))
        self.alphas = nn.Parameter(torch.tensor(alphas, dtype=torch.float32), requires_grad=False)
        self.risk_lambda = float(risk_lambda)
        self.m = float(ema_m)
        self.eps = float(eps)
        self.clamp = clamp
        self.use_tanh = bool(use_tanh)

    @torch.no_grad()
    def _ddp_reduce_mean_var(self, mean: torch.Tensor, var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            dist.all_reduce(mean, op=dist.ReduceOp.SUM)
            dist.all_reduce(var, op=dist.ReduceOp.SUM)
            mean = mean / world
            var = var / world
        return mean, var

    @torch.no_grad()
    def update_stats(self, s_bgk: torch.Tensor) -> None:
        # s_bgk: [B,G,K] on some device
        # 记录上一次用于 RSD
        self._prev_mu.copy_(self.mu)
        self._prev_sigma.copy_(torch.sqrt(self.var + self.eps))

        mean = s_bgk.mean(dim=(0, 1))  # [K]
        var = s_bgk.var(dim=(0, 1), unbiased=False)  # [K]
        mean, var = self._ddp_reduce_mean_var(mean.detach(), var.detach())

        # EMA 原地更新，保持 buffer 身份
        self.mu.mul_(self.m).add_(mean * (1.0 - self.m))
        self.var.mul_(self.m).add_(var * (1.0 - self.m))

    def forward(self, s_bgk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 将 alphas 放到正确 device/dtype
        if self.alphas.device != s_bgk.device:
            self.alphas.data = self.alphas.data.to(s_bgk.device)

        self.update_stats(s_bgk)
        sigma = torch.sqrt(self.var + self.eps)  # [K]
        z = (s_bgk - self.mu) / sigma  # [B,G,K]
        fused = (z * self.alphas).sum(dim=-1)  # [B,G]
        rm_var = z.var(dim=-1, unbiased=False)  # [B,G]
        fused = fused - self.risk_lambda * rm_var
        if self.clamp is not None:
            fused = fused.clamp(-self.clamp, self.clamp)
        if self.use_tanh:
            fused = torch.tanh(fused)
        return fused, rm_var, z

    @torch.no_grad()
    def rsd(self) -> torch.Tensor:
        """返回 RSD 指标：sum_k(|Δμ_k| + |Δσ_k|)。"""
        sigma = torch.sqrt(self.var + self.eps)
        return (self.mu - self._prev_mu).abs().sum() + (sigma - self._prev_sigma).abs().sum()


def group_dpo_loss(
    seq_logp_new: torch.Tensor,  # [B,G]
    seq_logp_ref: torch.Tensor,  # [B,G]
    rewards_bg: torch.Tensor,    # [B,G]
    *,
    beta_rank: float = 0.1,
    topk: int = 1,
    pair_cap: int = 2048,
) -> Tuple[torch.Tensor, int]:
    B, G = seq_logp_new.shape
    k = min(int(topk), G)
    top_vals, top_idx = rewards_bg.topk(k=k, dim=1)

    pairs: List[Tuple[int, int, int]] = []
    for b in range(B):
        winners = set(top_idx[b].tolist())
        for w in winners:
            for l in range(G):
                if l not in winners:
                    pairs.append((b, w, l))
    if len(pairs) == 0:
        return seq_logp_new.new_zeros(()), 0
    if len(pairs) > pair_cap:
        pairs = random.sample(pairs, pair_cap)

    device = seq_logp_new.device
    dtype = seq_logp_new.dtype
    idx = torch.tensor(pairs, device=device, dtype=torch.long)
    b_idx, w_idx, l_idx = idx[:, 0], idx[:, 1], idx[:, 2]

    wn = seq_logp_new[b_idx, w_idx]
    ln = seq_logp_new[b_idx, l_idx]
    wr = seq_logp_ref[b_idx, w_idx]
    lr = seq_logp_ref[b_idx, l_idx]

    dpo_arg = beta_rank * ((wn - wr) - (ln - lr))
    loss = F.softplus(-dpo_arg).mean()
    return loss.to(dtype), len(pairs)


def grpo_rl_loss(
    logp_new_bg: torch.Tensor,   # [B,G]
    logp_ref_bg: torch.Tensor,   # [B,G]
    fused_bg: torch.Tensor,      # [B,G]
    *,
    mode: str = "adv",
    tau: float = 0.4,
    clip_c: float = 1.5,
    baseline: str = "group_mean",
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    with torch.no_grad():
        if baseline == "group_mean":
            b = fused_bg.mean(dim=1, keepdim=True)
        else:
            b = fused_bg.mean()
        if mode == "soft":
            w = torch.exp((fused_bg - b) / tau)
        else:
            x = fused_bg - fused_bg.mean()
            x = x / (fused_bg.std(unbiased=False) + 1e-6)
            x = x.clamp(-clip_c, clip_c)
            w = torch.exp(x)
        ess = (w.sum() ** 2) / (w.pow(2).sum() + 1e-8)

    pg_loss = -(w.detach() * logp_new_bg).mean()
    approx_kl = (logp_new_bg - logp_ref_bg).mean()
    return pg_loss, approx_kl, float(ess.detach().cpu().item())


@torch.no_grad()
def estimate_true_kl_per_token(
    tok_logp_new: torch.Tensor,  # [B,G,T]
    tok_logp_ref: torch.Tensor,  # [B,G,T]
    lengths: torch.Tensor,       # [B,G]
) -> float:
    diff = tok_logp_new - tok_logp_ref  # [B,G,T]
    T = diff.size(-1)
    device = diff.device
    rng = torch.arange(T, device=device)[None, None, :]
    mask = rng < lengths[..., None]
    s = (diff * mask).sum(dim=-1)  # [B,G]
    n = lengths.clamp_min(1)
    kl_tok = (s / n).mean()
    return float(kl_tok.detach().cpu().item())


class KLController:
    def __init__(
        self,
        target: float = 0.02,
        up: float = 1.5,
        down: float = 0.67,
        beta_min: float = 1e-4,
        beta_max: float = 1.0,
    ) -> None:
        self.target = float(target)
        self.up = float(up)
        self.down = float(down)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def step(self, kl_token: float, beta: float) -> float:
        if kl_token > 1.5 * self.target:
            beta = min(beta * self.up, self.beta_max)
        elif kl_token < 0.5 * self.target:
            beta = max(beta * self.down, self.beta_min)
        return beta

