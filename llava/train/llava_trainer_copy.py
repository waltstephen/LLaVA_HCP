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
    grpo_stable_loss,
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
        mode      = getattr(self.args, "grpo_mode",  "soft")     # "soft"|"adv"
        tau       = getattr(self.args, "grpo_tau",   0.15)
        clip_w    = getattr(self.args, "grpo_clip",  5.0)
        kl_beta   = getattr(self.args, "kl_beta",    0.02)

        device = next(model.parameters()).device
        dtype  = next(model.parameters()).dtype
        pil_tf = transforms.ToPILImage()

        # ---------- 2. reference policy (懒加载一次) ----------
        if kl_beta > 0 and not hasattr(self, "_ref_model"):
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
        generations = generate_group_responses(
            model, gen_inputs, num_return_sequences=G
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
            token_logp_ref = (
                compute_group_token_log_probs(
                    self._ref_model,
                    generations,
                    inputs["input_ids"],
                    inputs.get("attention_mask", None),
                    images_for_logp,
                )
                if kl_beta > 0 else None
            )

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

        with torch.no_grad():
            raw_scores = []
            for rm in reward_models:
                sc = rm.score(images_reward, texts).to(device)  # [B*G]
                raw_scores.append(sc)

            clip_s, blip_s, dist_s = raw_scores
            dist_s = (dist_s - dist_s.min()) / (dist_s.max() - dist_s.min() + 1e-6)

            w_clip, w_blip, w_dist = 0.6, 0.3, 0.0
            combined = w_clip * clip_s + w_blip * blip_s + w_dist * dist_s
            rewards = combined.view(B, G).to(dtype=torch.float32)  # [B,G]

        # ---------- 8. Stable-GRPO + KL ----------
        loss, pol_loss, kl_loss = grpo_stable_loss(
            token_logp_new=token_logp_new,
            rewards=rewards,
            token_logp_ref=token_logp_ref,
            mode=mode,
            tau=tau,
            clip_w=clip_w,
            kl_beta=kl_beta,
        )

        # ---------- 9. 日志 ----------
        log_dict = {
            "policy_loss": pol_loss.item(),
            "kl_loss":     kl_loss.item(),
            "reward_mean": rewards.mean().item(),
            "rm0_clip":    clip_s.mean().item(),
            "rm1_blip":    blip_s.mean().item(),
            "rm2_dist":    dist_s.mean().item(),
        }
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

            # c) wandb（可选）
            if "wandb" in (self.args.report_to or []):
                try:
                    import wandb
                    wandb.log(
                        {"samples": wandb.Html(samples.replace("\n", "<br>"))},
                        step=self.state.global_step,
                        commit=False,
                    )
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
