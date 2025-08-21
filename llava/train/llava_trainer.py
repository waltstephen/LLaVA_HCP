import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
import copy
from torchvision import transforms
from typing import Tuple, Dict, Any, List

from rl_utils import (                               # 前文完整实现
    RewardModelCLIP,
    RewardModelBLIP_ITR,
    RewardModelDistinctiveness,
    generate_group_responses,
    compute_group_token_log_probs,
    # CaRPO 组件
    CalibratedRewardAggregator,
    KLController,
    group_dpo_loss,
    grpo_rl_loss,
    estimate_true_kl_per_token,
    compute_group_log_probs,
)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

def _model_device(model: torch.nn.Module) -> torch.device:
    """
    返回模型所在 device：
      • 普通 nn.Module            → model.device
      • DataParallel / DDP 包装器 → next(model.parameters()).device
    """
    return getattr(model, "device", None) or next(model.parameters()).device
class LLaVATrainer(Trainer):

    ########################################################################
    #                         🚩  RL-GRPO compute_loss                     #
    ########################################################################

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # ---------- 0. 纯 SFT 分支 ----------
        if not getattr(self.args, "rl_grpo", False):
            return super().compute_loss(model, inputs, return_outputs)

        # ---------- 1. 超参 ----------
        G         = getattr(self.args, "rl_group_size", 4)
        # CaRPO / PR-MD 相关
        mode      = getattr(self.args, "grpo_mode",  "adv")
        tau       = getattr(self.args, "grpo_tau",   0.4)
        clip_c    = getattr(self.args, "grpo_clip",  1.5)
        # Rank-first
        beta_rank = getattr(self.args, "beta_rank",  0.1)
        pair_topk = getattr(self.args, "pair_topk",  1)
        pair_cap  = getattr(self.args, "pair_cap",   2048)
        # Aggregator
        reward_alphas = torch.tensor(getattr(self.args, "reward_alphas", [0.6, 0.3, 0.1]), dtype=torch.float32)
        risk_lambda   = getattr(self.args, "risk_lambda", 0.2)
        ema_m         = getattr(self.args, "ema_m", 0.95)
        agg_clamp     = getattr(self.args, "agg_clamp", None)
        agg_use_tanh  = getattr(self.args, "agg_use_tanh", False)
        # Penalty（加性）
        short_threshold = getattr(self.args, "short_threshold", 8)
        lambda_empty    = getattr(self.args, "lambda_empty", 1.0)
        lambda_len      = getattr(self.args, "lambda_len", 0.05)
        # KL 控制器
        target_kl_token = getattr(self.args, "target_kl_token", 0.02)
        kl_beta_min     = getattr(self.args, "kl_beta_min", 1e-4)
        kl_beta_max     = getattr(self.args, "kl_beta_max", 0.5)
        # Loss 组合
        rank_warmup_steps = getattr(self.args, "rank_warmup_steps", 800)
        rank_weight       = getattr(self.args, "rank_weight", 0.5)
        rl_weight         = getattr(self.args, "rl_weight", 0.5)

        device = next(model.parameters()).device
        dtype  = next(model.parameters()).dtype
        pil_tf = transforms.ToPILImage()

        # ---------- 2. reference policy (懒加载一次) ----------
        # 参考策略懒加载（需要 KL 正则 或 Rank-first DPO 的 reference）
        if not hasattr(self, "_ref_model"):
            self._ref_model = copy.deepcopy(model if not hasattr(model, "module")
                                            else model.module).eval()
            for p in self._ref_model.parameters():
                p.requires_grad_(False)

        # ---------- 3. 懒加载奖励模型 ----------
        if not hasattr(self, "_reward_models"):
            self._reward_models = [
                RewardModelCLIP(),
                RewardModelBLIP_ITR(),
                RewardModelDistinctiveness(group_size=G),
            ]
        reward_models = self._reward_models

        # 奖励聚合器（跨步 EMA）
        if not hasattr(self, "_reward_aggregator"):
            self._reward_aggregator = CalibratedRewardAggregator(
                K=len(reward_models),
                alphas=reward_alphas.tolist(),
                risk_lambda=risk_lambda,
                ema_m=ema_m,
                clamp=agg_clamp,
                use_tanh=agg_use_tanh,
            ).to(device)

        # KL 控制器（持久化 beta）
        if not hasattr(self, "_kl_controller"):
            self._kl_controller = KLController(
                target=target_kl_token,
                beta_min=kl_beta_min,
                beta_max=kl_beta_max,
            )
        if not hasattr(self, "_kl_beta_value"):
            self._kl_beta_value = float(getattr(self.args, "kl_beta_init", 0.05))

        # ---------- 4. 处理 images ----------
        if "images" not in inputs:
            raise ValueError("batch 缺少 `images` 字段")

        gen_inputs = dict(inputs)                       # 浅拷贝给 generate
        if isinstance(inputs["images"], torch.Tensor):  # Tensor[B,3,H,W]
            img_tensor = inputs["images"].to(device, dtype=dtype)
            gen_inputs["images"] = img_tensor
            base_pils       = [pil_tf(t.cpu()) for t in inputs["images"]]
            images_for_logp = img_tensor                 # [B,3,H,W]
        else:                                           # List[PIL]
            base_pils       = inputs["images"]
            images_for_logp = base_pils

        # ---------- 5. 生成组回复 ----------
        # 传入温度控制（允许 CLI 控制），并保留其它默认采样超参
        gen_kwargs = {"temperature": float(getattr(self.args, "temperature", 1.0))}
        generations = generate_group_responses(
            model, gen_inputs, num_return_sequences=G, generation_kwargs=gen_kwargs
        )                                               # [B,G,L]
        B, _, L = generations.shape

        # ---------- 6. token-level log π_new / π_ref ----------
        token_logp_new = compute_group_token_log_probs(
            model,
            generations,
            inputs["input_ids"],
            inputs.get("attention_mask", None),
            images_for_logp,
        )                                               # [B,G,L]

        with torch.no_grad():
            token_logp_ref = compute_group_token_log_probs(
                self._ref_model,
                generations,
                inputs["input_ids"],
                inputs.get("attention_mask", None),
                images_for_logp,
            )

        # 句级 logp（仅生成段）
        # seq_logp_new 需要梯度，不能放在 no_grad 中
        seq_logp_new = compute_group_log_probs(
            model, generations, inputs["input_ids"], inputs.get("attention_mask", None), images_for_logp
        )  # [B,G]
        
        with torch.no_grad():
            seq_logp_ref = compute_group_log_probs(
                self._ref_model, generations, inputs["input_ids"], inputs.get("attention_mask", None), images_for_logp
            )  # [B,G]

        # ---------- 7. 奖励打分 ----------
        # 让图片批次与文本批次同维度 (B*G)
        if isinstance(images_for_logp, torch.Tensor):
            images_reward = images_for_logp.repeat_interleave(G, 0)
        else:
            images_reward = [img for img in images_for_logp for _ in range(G)]

        texts = self.tokenizer.batch_decode(
            generations.view(B * G, L),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # ---------- 7. 奖励打分与校准融合（CaRPO） ----------
        with torch.no_grad():
            raw_scores = []
            for rm in reward_models:
                sc = rm.score(images_reward, texts).to(device)  # [B*G]
                raw_scores.append(sc)

            # 堆叠为 [B,G,K]
            s_bgk = torch.stack(raw_scores, dim=-1).view(B, G, -1).to(device=device, dtype=torch.float32)
            fused, rm_var, z_bgk = self._reward_aggregator(s_bgk)  # [B,G], [B,G], [B,G,K]

            # 加性惩罚：空/短
            if short_threshold is not None and (lambda_empty > 0 or lambda_len > 0):
                penalties = []
                for s in texts:
                    st = s.strip()
                    if len(st) == 0:
                        penalties.append(lambda_empty)
                    elif len(st) < short_threshold:
                        penalties.append(lambda_len * float(short_threshold - len(st)))
                    else:
                        penalties.append(0.0)
                penalties = torch.tensor(penalties, device=fused.device, dtype=fused.dtype).view(B, G)
                fused = fused - penalties

            rewards = fused

        # ---------- 8. Rank-first：Group-DPO ----------
        rank_loss, num_pairs = group_dpo_loss(
            seq_logp_new=seq_logp_new,
            seq_logp_ref=seq_logp_ref,
            rewards_bg=rewards,
            beta_rank=beta_rank,
            topk=pair_topk,
            pair_cap=pair_cap,
        )

        # ---------- 9. PR-MD：RL + 受控 KL ----------
        pg_loss, approx_kl, ess = grpo_rl_loss(
            logp_new_bg=seq_logp_new,
            logp_ref_bg=seq_logp_ref,
            fused_bg=rewards,
            mode=mode,
            tau=tau,
            clip_c=clip_c,
            baseline="group_mean",
        )

        # True KL/token 估计与 β 自适应
        gen_lengths = torch.full((B, G), token_logp_new.size(-1), device=device, dtype=torch.long)
        kl_token = estimate_true_kl_per_token(token_logp_new, token_logp_ref, gen_lengths)
        self._kl_beta_value = self._kl_controller.step(kl_token, self._kl_beta_value)

        # ---------- 9.1 ESS 自适应稳定性控制：根据当前 ESS 调整温度 ----------
        try:
            B = int(generations.size(0))
            ess_target_ratio = float(getattr(self.args, "ess_target_ratio", 0.6))
            ess_tol = float(getattr(self.args, "ess_tolerance", 0.1))
            temp = float(getattr(self.args, "temperature", 1.0))
            t_min = float(getattr(self.args, "temperature_min", 0.3))
            t_max = float(getattr(self.args, "temperature_max", 1.5))
            t_up  = float(getattr(self.args, "temperature_up", 1.07))
            t_dn  = float(getattr(self.args, "temperature_down", 0.93))
            G_eff = int(getattr(self.args, "rl_group_size", G))
            ess_target = max(1.0, ess_target_ratio * (B * G_eff))
            if ess < (1.0 - ess_tol) * ess_target:
                temp = min(t_max, temp * t_up)
            elif ess > (1.0 + ess_tol) * ess_target:
                temp = max(t_min, temp * t_dn)
            # 将更新后的温度写回，影响后续 step 的生成
            setattr(self.args, "temperature", float(temp))
        except Exception:
            pass

        loss_rl = pg_loss + self._kl_beta_value * approx_kl

        # 合成总损失（先排后推、支持 warmup）
        step = int(getattr(self.state, "global_step", 0))
        if step < rank_warmup_steps:
            loss = rank_loss
        else:
            loss = rank_weight * rank_loss + rl_weight * loss_rl

        # ---------- 9. 日志 ----------
        # ---------- 10. 日志 ----------
        with torch.no_grad():
            # KCE
            kce = abs(kl_token - target_kl_token)
            # RSD
            rsd_val = float(self._reward_aggregator.rsd().detach().cpu().item())
            # IRC（平均 Spearman，近似实现）
            try:
                z_flat = z_bgk.view(B * G, -1)  # [B·G,K]
                ranks = z_flat.argsort(dim=0).argsort(dim=0).to(dtype=torch.float32)
                ranks = (ranks - ranks.mean(dim=0, keepdim=True)) / (ranks.std(dim=0, keepdim=True) + 1e-6)
                corr_mat = (ranks.t() @ ranks) / (ranks.size(0) - 1)
                K = corr_mat.size(0)
                irc = (corr_mat.sum() - torch.diag(corr_mat).sum()) / (K * (K - 1))
                irc_val = float(irc.detach().cpu().item())
            except Exception:
                irc_val = 0.0
            rm_var_mean = float(rm_var.mean().detach().cpu().item())

        # 记录 RM 原始分数的统计（均值）与 fused 的统计
        with torch.no_grad():
            try:
                rm_means = s_bgk.mean(dim=(0, 1))  # [K]
                fused_mean = rewards.mean()
                rm_logs = {f"rm{k}_mean": float(rm_means[k].detach().cpu().item()) for k in range(rm_means.numel())}
            except Exception:
                rm_logs = {}
                fused_mean = float("nan")

        log_dict = {
            "loss_total": float(loss.detach().cpu().item()),
            "loss_rank": float(rank_loss.detach().cpu().item()),
            "loss_pg": float(pg_loss.detach().cpu().item()),
            "approx_kl": float(approx_kl.detach().cpu().item()),
            "kl_beta": float(self._kl_beta_value),
            "kl_token": float(kl_token),
            "kce": float(kce),
            "ess": float(ess),
            "temperature": float(getattr(self.args, "temperature", 1.0)),
            "fused_reward_mean": float(fused_mean),
            "num_pairs": int(num_pairs),
            "rm_var_mean": rm_var_mean,
            "rsd": rsd_val,
            "irc": irc_val,
        }
        # 合并 RM 均值日志
        try:
            log_dict.update(rm_logs)
        except Exception:
            pass
        self.log(log_dict)

        # ---------- 9.1 额外可读样本日志 ----------
        # 只在 rank 0 且到达指定间隔时写一次
        if (self.args.sample_log_freq > 0
            and self.state.global_step % self.args.sample_log_freq == 0
            and self.is_world_process_zero()  # 等价 local_rank in (-1,0)
        ):
            show_k = min(len(texts), self.args.sample_log_n)
            tag = f"[step {self.state.global_step:06d}]"
            samples = "\n".join(
                f"{tag} {i}: {texts[i][:300]}" for i in range(show_k)
            )

            # a) 终端
            print(samples, flush=True)

            # b) 追加到文件
            path = os.path.join(self.args.output_dir, "sample_log.txt")
            with open(path, "a", encoding="utf-8") as f:
                f.write(samples + "\n")

            # c) swanlab（可选）
            if "swanlab" in (self.args.report_to or []):
                try:
                    import swanlab
                    # 以纯文本形式记录可读样本
                    swanlab.log({"samples": samples}, step=self.state.global_step)
                except ImportError:
                    pass


        return (loss, None) if return_outputs else loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
