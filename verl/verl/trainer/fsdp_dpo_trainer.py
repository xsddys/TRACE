# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import torch
import torch.distributed
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import DPODataset, DefenseDPODataset, DefenseSFTDataset, SFTDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking


def convert_to_regular_types(obj):
    from omegaconf import DictConfig, ListConfig

    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    if isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


class FSDPDPOTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        helpful_train_dataset: Dataset | None = None,
        helpful_val_dataset: Dataset | None = None,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.tokenizer = tokenizer
        self.helpful_train_dataset = helpful_train_dataset
        self.helpful_val_dataset = helpful_val_dataset

        self._normalize_config_bsz()
        self._build_dataloader(train_dataset, val_dataset)
        self._build_models_optimizer()

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size()
        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        self.config.data.train_batch_size //= dp_size
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0
        if self.config.helpful_data.enabled:
            assert (
                self.config.helpful_data.train_batch_size % dp_size == 0
            ), f"Helpful global batch size {self.config.helpful_data.train_batch_size} is not divisible by dp size {dp_size}"
            self.config.helpful_data.train_batch_size //= dp_size
            assert self.config.helpful_data.train_batch_size % self.config.helpful_data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.helpful_train_sampler = None
        self.helpful_train_dataloader = None
        self.helpful_val_sampler = None
        self.helpful_val_dataloader = None

        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        train_num_workers = int(self.config.data.get("num_workers", 0))
        val_num_workers = int(self.config.data.get("val_num_workers", train_num_workers))

        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=train_num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=False)
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=val_num_workers,
            pin_memory=True,
            drop_last=False,
        )

        if self.config.helpful_data.enabled:
            helpful_train_num_workers = int(self.config.helpful_data.get("num_workers", train_num_workers))
            helpful_val_num_workers = int(self.config.helpful_data.get("val_num_workers", helpful_train_num_workers))

            self.helpful_train_sampler = DistributedSampler(
                self.helpful_train_dataset,
                shuffle=True,
                num_replicas=world_size,
                rank=rank,
                drop_last=True,
            )
            self.helpful_train_dataloader = DataLoader(
                dataset=self.helpful_train_dataset,
                batch_size=self.config.helpful_data.train_batch_size,
                sampler=self.helpful_train_sampler,
                num_workers=helpful_train_num_workers,
                pin_memory=True,
                drop_last=True,
            )

            self.helpful_val_sampler = DistributedSampler(
                self.helpful_val_dataset,
                shuffle=False,
                num_replicas=world_size,
                rank=rank,
                drop_last=False,
            )
            self.helpful_val_dataloader = DataLoader(
                dataset=self.helpful_val_dataset,
                batch_size=self.config.helpful_data.micro_batch_size_per_gpu,
                sampler=self.helpful_val_sampler,
                num_workers=helpful_val_num_workers,
                pin_memory=True,
                drop_last=False,
            )

    def _build_single_model(self, model_path: str, trainable: bool):
        local_model_path = copy_to_local(src=model_path, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            import importlib

            importlib.import_module(self.config.model.external_lib)

        trust_remote_code = self.config.model.trust_remote_code
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh)

        torch_dtype = torch.float32 if trainable else torch.bfloat16
        with init_context():
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=model)

            if trainable and self.config.model.get("lora_rank", 0) > 0:
                model.enable_input_require_grads()
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "lora_dropout": self.config.model.get("lora_dropout", 0.0),
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                model = get_peft_model(model, LoraConfig(**lora_config))

        if trainable and self.config.model.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if not trainable:
            for parameter in model.parameters():
                parameter.requires_grad = False

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
        auto_wrap_policy = get_fsdp_wrap_policy(
            model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=trainable and self.config.model.get("lora_rank", 0) > 0,
        )

        cpu_offload = None
        if self.config.model.fsdp_config.cpu_offload:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_model = FSDP(
            module=model,
            auto_wrap_policy=auto_wrap_policy,
            param_init_fn=init_fn,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True,
            device_id=torch.cuda.current_device(),
            cpu_offload=cpu_offload,
            use_orig_params=False,
        )

        return model, fsdp_model

    def _build_models_optimizer(self):
        log_gpu_memory_usage("Before DPO model allocation")

        self.policy_model, self.policy_fsdp_model = self._build_single_model(
            model_path=self.config.model.partial_pretrain,
            trainable=True,
        )
        ref_model_path = self.config.model.get("ref_model_path", None) or self.config.model.partial_pretrain
        self.ref_model, self.ref_fsdp_model = self._build_single_model(
            model_path=ref_model_path,
            trainable=False,
        )

        self.optimizer = optim.AdamW(
            self.policy_fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps,
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

        log_gpu_memory_usage("After DPO model allocation")

    def _iter_micro_batches(self, batch: dict[str, torch.Tensor], micro_batch_size: int):
        batch_size = next(iter(batch.values())).size(0)
        for start in range(0, batch_size, micro_batch_size):
            end = start + micro_batch_size
            yield {key: value[start:end].cuda(non_blocking=True) for key, value in batch.items()}

    def _compute_sequence_logps(self, model: FSDP, prefix: str, micro_batch: dict[str, torch.Tensor]):
        input_ids = micro_batch[f"{prefix}_input_ids"]
        attention_mask = micro_batch[f"{prefix}_attention_mask"]
        position_ids = micro_batch[f"{prefix}_position_ids"]
        loss_mask = micro_batch[f"{prefix}_loss_mask"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_loss_mask = loss_mask[:, :-1].contiguous().to(shift_logits.dtype)

        token_logps = torch.gather(F.log_softmax(shift_logits.float(), dim=-1), dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        sequence_logps = (token_logps * shift_loss_mask).sum(dim=-1)

        if self.config.dpo.average_log_prob:
            token_counts = shift_loss_mask.sum(dim=-1).clamp_min(1.0)
            sequence_logps = sequence_logps / token_counts

        return sequence_logps

    def _compute_micro_batch_metrics(self, micro_batch: dict[str, torch.Tensor]):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            policy_chosen_logps = self._compute_sequence_logps(self.policy_fsdp_model, "chosen", micro_batch)
            policy_rejected_logps = self._compute_sequence_logps(self.policy_fsdp_model, "rejected", micro_batch)

        if self.config.dpo.reference_free:
            ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
            ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
        else:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ref_chosen_logps = self._compute_sequence_logps(self.ref_fsdp_model, "chosen", micro_batch)
                ref_rejected_logps = self._compute_sequence_logps(self.ref_fsdp_model, "rejected", micro_batch)

        policy_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = self.config.dpo.beta * (policy_logratios - ref_logratios)

        loss_type = self.config.dpo.get("loss_type", "sigmoid")
        if loss_type != "sigmoid":
            raise ValueError(f"Unsupported DPO loss_type: {loss_type}")

        label_smoothing = self.config.dpo.label_smoothing
        losses = -(1.0 - label_smoothing) * F.logsigmoid(logits) - label_smoothing * F.logsigmoid(-logits)
        chosen_rewards = self.config.dpo.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.config.dpo.beta * (policy_rejected_logps - ref_rejected_logps)

        metrics = {
            "loss": losses.mean().detach(),
            "reward_accuracy": (chosen_rewards > rejected_rewards).float().mean().detach(),
            "chosen_reward": chosen_rewards.mean().detach(),
            "rejected_reward": rejected_rewards.mean().detach(),
            "policy_logratio": policy_logratios.mean().detach(),
            "ref_logratio": ref_logratios.mean().detach(),
            "margin": (chosen_rewards - rejected_rewards).mean().detach(),
        }
        return losses.mean(), metrics

    def _compute_sft_micro_batch_loss(self, micro_batch: dict[str, torch.Tensor]):
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        loss_mask = micro_batch["loss_mask"][:, :-1].contiguous().to(torch.float32)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.policy_fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            token_losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.reshape(-1),
                reduction="none",
            ).view_as(shift_labels)
            token_losses = token_losses * loss_mask.to(token_losses.device)
            denom = loss_mask.sum().clamp_min(1.0)
            loss = token_losses.sum() / denom
        return loss, {"sft_loss": loss.detach()}

    def _reduce_metrics(self, metrics: dict[str, torch.Tensor], prefix: str):
        reduced = {}
        for key, value in metrics.items():
            tensor_value = value.detach().float()
            torch.distributed.all_reduce(tensor_value, op=torch.distributed.ReduceOp.AVG)
            reduced[f"{prefix}/{key}"] = tensor_value.item()
        return reduced

    def training_step(self, batch: dict[str, torch.Tensor], helpful_batch: dict[str, torch.Tensor] | None = None):
        self.policy_fsdp_model.train()
        self.ref_fsdp_model.eval()
        self.optimizer.zero_grad()

        micro_batches = list(self._iter_micro_batches(batch, self.config.data.micro_batch_size_per_gpu))
        metric_sums = defaultdict(float)
        helpful_metric_sums = defaultdict(float)
        for micro_batch in micro_batches:
            loss, metrics = self._compute_micro_batch_metrics(micro_batch)
            (loss / len(micro_batches)).backward()
            micro_batch_size = next(iter(micro_batch.values())).size(0)
            for key, value in metrics.items():
                metric_sums[key] += value.item() * micro_batch_size

        if helpful_batch is not None:
            helpful_micro_batches = list(self._iter_micro_batches(helpful_batch, self.config.helpful_data.micro_batch_size_per_gpu))
            for micro_batch in helpful_micro_batches:
                helpful_loss, helpful_metrics = self._compute_sft_micro_batch_loss(micro_batch)
                (self.config.joint.sft_lambda * helpful_loss / len(helpful_micro_batches)).backward()
                helpful_micro_batch_size = next(iter(micro_batch.values())).size(0)
                for key, value in helpful_metrics.items():
                    helpful_metric_sums[key] += value.item() * helpful_micro_batch_size

        max_grad_norm = self.config.optim.get("max_grad_norm", self.config.optim.get("clip_grad", 1.0))
        grad_norm = self.policy_fsdp_model.clip_grad_norm_(max_norm=max_grad_norm)
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        self.lr_scheduler.step()

        batch_size = next(iter(batch.values())).size(0)
        metrics = {key: torch.tensor(value / batch_size, device="cuda") for key, value in metric_sums.items()}
        if helpful_batch is not None:
            helpful_batch_size = next(iter(helpful_batch.values())).size(0)
            for key, value in helpful_metric_sums.items():
                metrics[key] = torch.tensor(value / helpful_batch_size, device="cuda")
            if "sft_loss" in metrics:
                metrics["joint_loss"] = metrics["loss"] + self.config.joint.sft_lambda * metrics["sft_loss"]
        metrics["grad_norm"] = grad_norm.detach().float()
        metrics["lr"] = torch.tensor(self.lr_scheduler.get_last_lr()[0], device="cuda")
        return self._reduce_metrics(metrics, prefix="train")

    def validation_step(self, batch: dict[str, torch.Tensor]):
        self.policy_fsdp_model.eval()
        self.ref_fsdp_model.eval()

        metric_sums = defaultdict(float)
        sample_count = 0
        with torch.no_grad():
            for micro_batch in self._iter_micro_batches(batch, self.config.data.micro_batch_size_per_gpu):
                _, metrics = self._compute_micro_batch_metrics(micro_batch)
                micro_batch_size = next(iter(micro_batch.values())).size(0)
                sample_count += micro_batch_size
                for key, value in metrics.items():
                    metric_sums[key] += value.item() * micro_batch_size

        metrics = {key: torch.tensor(value / sample_count, device="cuda") for key, value in metric_sums.items()}
        return self._reduce_metrics(metrics, prefix="val")

    def helpful_validation_step(self, batch: dict[str, torch.Tensor]):
        self.policy_fsdp_model.eval()
        metric_sums = defaultdict(float)
        sample_count = 0
        with torch.no_grad():
            for micro_batch in self._iter_micro_batches(batch, self.config.helpful_data.micro_batch_size_per_gpu):
                loss, metrics = self._compute_sft_micro_batch_loss(micro_batch)
                micro_batch_size = next(iter(micro_batch.values())).size(0)
                sample_count += micro_batch_size
                for key, value in metrics.items():
                    metric_sums[key] += value.item() * micro_batch_size

        metrics = {key: torch.tensor(value / sample_count, device="cuda") for key, value in metric_sums.items()}
        return self._reduce_metrics(metrics, prefix="val_helpful")

    def save_checkpoint(self, step: int):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy_fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.policy_fsdp_model.state_dict()

        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.policy_model.save_pretrained(path, state_dict=state_dict)
            self.tokenizer.save_pretrained(path)
            if self.config.trainer.default_hdfs_dir:
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()
        tracking = None
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        evals_per_epoch = int(self.config.trainer.get("evals_per_epoch", 1))
        eval_interval = max(1, math.ceil(self.steps_per_epoch / max(1, evals_per_epoch)))
        helpful_train_iterator = None
        if self.config.helpful_data.enabled and self.helpful_train_dataloader is not None:
            helpful_train_iterator = iter(self.helpful_train_dataloader)

        global_step = 0
        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch)
            if self.config.helpful_data.enabled and self.helpful_train_sampler is not None:
                self.helpful_train_sampler.set_epoch(epoch)
                helpful_train_iterator = iter(self.helpful_train_dataloader)
            epoch_step = 0
            for batch in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
            ):
                global_step += 1
                epoch_step += 1
                helpful_batch = None
                if helpful_train_iterator is not None:
                    try:
                        helpful_batch = next(helpful_train_iterator)
                    except StopIteration:
                        helpful_train_iterator = iter(self.helpful_train_dataloader)
                        helpful_batch = next(helpful_train_iterator)
                metrics = self.training_step(batch, helpful_batch=helpful_batch)
                if rank == 0:
                    tracking.log(data=metrics, step=global_step)

                should_run_epoch_eval = (epoch_step % eval_interval == 0) or (epoch_step == self.steps_per_epoch)
                if should_run_epoch_eval:
                    val_metrics = self.evaluate()
                    if rank == 0:
                        tracking.log(data=val_metrics, step=global_step)
                    self.save_checkpoint(global_step)

                if global_step >= total_training_steps:
                    if not should_run_epoch_eval:
                        val_metrics = self.evaluate()
                        if rank == 0:
                            tracking.log(data=val_metrics, step=global_step)
                        self.save_checkpoint(global_step)
                    return

    def evaluate(self):
        if len(self.val_dataloader) == 0:
            return {}
        collected = defaultdict(list)
        for batch in self.val_dataloader:
            metrics = self.validation_step(batch)
            for key, value in metrics.items():
                collected[key].append(value)

        if self.config.helpful_data.enabled and self.helpful_val_dataloader is not None and len(self.helpful_val_dataloader) > 0:
            helpful_collected = defaultdict(list)
            for batch in self.helpful_val_dataloader:
                metrics = self.helpful_validation_step(batch)
                for key, value in metrics.items():
                    helpful_collected[key].append(value)
            for key, values in helpful_collected.items():
                collected[key].append(sum(values) / len(values))

        reduced = {key: sum(values) / len(values) for key, values in collected.items()}
        if self.config.helpful_data.enabled and "val/loss" in reduced and "val_helpful/sft_loss" in reduced:
            reduced["val/joint_loss"] = reduced["val/loss"] + self.config.joint.sft_lambda * reduced["val_helpful/sft_loss"]
        return reduced


@hydra.main(config_path="config", config_name="dpo_trainer", version_base=None)
@record
def main(config):
    _, _, world_size = initialize_global_process_group()
    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))

    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

    dataset_cls = config.data.get("dataset_cls", "dpo")
    if dataset_cls == "defense_dpo":
        dataset_type = DefenseDPODataset
    elif dataset_cls == "dpo":
        dataset_type = DPODataset
    else:
        raise ValueError(f"Unknown dataset_cls: {dataset_cls}")

    train_dataset = dataset_type(parquet_files=config.data.train_files, tokenizer=tokenizer, config=config.data)
    val_dataset = dataset_type(parquet_files=config.data.val_files, tokenizer=tokenizer, config=config.data)

    helpful_train_dataset = None
    helpful_val_dataset = None
    if config.helpful_data.enabled:
        helpful_dataset_cls = config.helpful_data.get("dataset_cls", "defense_sft")
        if helpful_dataset_cls == "defense_sft":
            helpful_dataset_type = DefenseSFTDataset
        elif helpful_dataset_cls == "sft":
            helpful_dataset_type = SFTDataset
        else:
            raise ValueError(f"Unknown helpful_data.dataset_cls: {helpful_dataset_cls}")
        helpful_train_dataset = helpful_dataset_type(parquet_files=config.helpful_data.train_files, tokenizer=tokenizer, config=config.helpful_data)
        helpful_val_dataset = helpful_dataset_type(parquet_files=config.helpful_data.val_files, tokenizer=tokenizer, config=config.helpful_data)

    trainer = FSDPDPOTrainer(
        config=config,
        device_mesh=device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        helpful_train_dataset=helpful_train_dataset,
        helpful_val_dataset=helpful_val_dataset,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
