"""Train one model from the Hydra config (conf/config.yaml), on one device or under torchrun.

    torchrun --standalone --nproc_per_node=8 scripts/train.py +experiment=<name> system.data_root=<dir>

The paper runs used 8 GPUs (micro batch 6, global batch 96). The world size decides each
rank's data stream, so training.world_size (default 8) refuses any other; pass
training.world_size=null for a smoke run on fewer devices.
"""

import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict

import hydra
import numpy as np
import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.distributed import ReduceOp, all_reduce, destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import trange

from modeling.models.model import config_args
from training import (
    CheckpointManager,
    FixedRandomChunkDistributedSampler,
    TokenDataset,
    final_checkpoint_path,
    fresh_start_offset,
    prepare_wandb_dir_from_config,
)


def _format_tokens(count: int) -> str:
    units = ["", "K", "M", "B", "T"]
    value = float(count)
    unit = 0
    while abs(value) >= 1000.0 and unit < len(units) - 1:
        value /= 1000.0
        unit += 1
    if unit == 0:
        return f"{int(value)}"
    return f"{value:.2f}{units[unit]}"


def _accumulate_stats(stats_dict, stats_sums, stats_occurrences):
    for key, value in stats_dict.items():
        scalar_value = float(value.item()) if torch.is_tensor(value) else float(value)
        stats_sums[key] = stats_sums.get(key, 0.0) + scalar_value
        stats_occurrences[key] = stats_occurrences.get(key, 0) + 1


def _summarize_stats(stats_sums, stats_occurrences):
    """Mean of each plain stat; `<x>_sum` / `<x>_count` pairs become their ratio `<x>`."""
    metrics = {}
    for key, total in stats_sums.items():
        if key.endswith("_sum") or key.endswith("_count"):
            continue
        count = stats_occurrences.get(key, 0)
        if count > 0:
            metrics[key] = total / count
    for key, total_sum in stats_sums.items():
        if not key.endswith("_sum"):
            continue
        base = key[:-4]
        total_count = stats_sums.get(f"{base}_count", 0.0)
        if total_count > 0:
            metrics[base] = total_sum / total_count
    if "token_nll" in metrics:
        metrics["token_ppl"] = math.exp(metrics["token_nll"])
    return metrics


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    t = cfg.training
    if int(os.environ.get('RANK', -1)) <= 0:
        print(OmegaConf.to_yaml(cfg))
    if not cfg.system.data_root:
        raise ValueError("cfg.system.data_root is required and must be a non-empty path")
    if cfg.system.dtype not in ("float32", "bfloat16"):
        raise ValueError(f"system.dtype must be float32 or bfloat16, got {cfg.system.dtype!r}")
    if t.sampler_resume not in ("legacy", "per_rank"):
        raise ValueError(f"training.sampler_resume must be 'legacy' or 'per_rank', got {t.sampler_resume!r}")

    # Checkpoints go to <out_root>/out/<run>; the corpus is read from <data_root>/data/<dataset>.
    out_dir = os.path.join(cfg.system.out_root, "out", cfg.logging.wandb_run_name)

    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=cfg.system.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        print(f"DDP rank: {ddp_rank}, local rank: {ddp_local_rank}, world size: {ddp_world_size}, device: {device}")
        torch.cuda.set_device(device)
    else:
        ddp_rank = ddp_local_rank = 0
        ddp_world_size = 1
        device = cfg.system.device
    master_process = ddp_rank == 0
    if t.world_size is not None and t.world_size != ddp_world_size:
        raise ValueError(f"training.world_size={t.world_size} but running at world size {ddp_world_size}")

    assert t.global_batch_size % (t.micro_batch_size * ddp_world_size) == 0, \
        f"global_batch_size ({t.global_batch_size}) must be divisible by (micro_batch_size ({t.micro_batch_size}) * num_gpus ({ddp_world_size}))"
    gradient_accumulation_steps = t.global_batch_size // (t.micro_batch_size * ddp_world_size)

    base_seed = int(t.seed)
    torch.manual_seed(base_seed + ddp_rank)
    random.seed(base_seed + ddp_rank)
    if cfg.system.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        if master_process:
            print("Deterministic mode enabled")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[cfg.system.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Validate the token-based eval and LR schedules.
    if t.eval_interval_tokens is None or t.eval_interval_tokens <= 0:
        raise ValueError(f"training.eval_interval_tokens must be > 0, got {t.eval_interval_tokens}")
    if t.eval_total_tokens is None or t.eval_total_tokens <= 0:
        raise ValueError(f"training.eval_total_tokens must be > 0, got {t.eval_total_tokens}")
    sched = cfg.scheduler
    assert sched.warmup_tokens >= 0, f"warmup_tokens must be >= 0, got {sched.warmup_tokens}"
    assert sched.lr_decay_tokens > 0, f"lr_decay_tokens must be > 0, got {sched.lr_decay_tokens}"
    assert sched.warmup_tokens + sched.lr_decay_tokens <= t.max_tokens, \
        f"warmup_tokens ({sched.warmup_tokens:,}) + lr_decay_tokens ({sched.lr_decay_tokens:,}) " \
        f"exceeds max_tokens ({t.max_tokens:,})"
    decay_start_tokens = t.max_tokens - sched.lr_decay_tokens

    # Resume resolution prefers the rolling ckpt.pt pre-decay and the latest named
    # checkpoint once in the decay phase, so the manager needs decay_start_tokens now.
    checkpoint_manager = CheckpointManager(
        out_dir=out_dir,
        save_every=t.save_every,
        master_process=master_process,
        decay_start_tokens=decay_start_tokens if sched.decay_lr else None,
        rolling_save_every=t.rolling_save_every,
        world_size=ddp_world_size,
    )

    checkpoint = None
    if t.init_from == 'resume':
        checkpoint_path = checkpoint_manager.resolve_checkpoint_path(t.resume_checkpoint)
        if checkpoint_path is None:
            print(f"No checkpoint found in {out_dir}, starting from scratch instead")
        else:
            print(f"Resuming training from {out_dir}")
            checkpoint = checkpoint_manager.load_checkpoint(device, checkpoint_path=checkpoint_path)
            OmegaConf.set_struct(cfg, False)
            config_cls = get_class(cfg.model.config._target_)
            for k, v in config_args(config_cls, checkpoint['model_args']).items():
                cfg.model.config[k] = v
            OmegaConf.set_struct(cfg, True)
    elif t.init_from != 'scratch':
        raise ValueError(f"Unknown init_from: {t.init_from}")

    model = instantiate(cfg.model)
    print("Model instantiated:", type(model).__name__)
    print("Model config:", model.config)
    iter_num = 0
    best_val_loss = 1e9
    wandb_run_id = None  # from the checkpoint, else from wandb.init
    next_eval_tokens = None
    if checkpoint is None:
        model_args = asdict(model.config)
    else:
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        state_dict.pop("freqs_cis", None)
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print("Checkpoint load with non-matching keys.")
            print(f"   Missing keys: {load_result.missing_keys}")
            print(f"   Unexpected keys: {load_result.unexpected_keys}")
        model_args = checkpoint['model_args']
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        wandb_run_id = checkpoint.get('wandb_run_id')
        next_eval_tokens = int(checkpoint['next_eval_tokens'])
        if master_process:
            print(f"Resume state: checkpoint={checkpoint_path}, iter_num={iter_num}, "
                  f"last_save_tokens={int(checkpoint.get('last_save_tokens', iter_num)):,}")

    # Freeze-and-retrain: load the STATE tower from a trained checkpoint and set
    # requires_grad=False on it BEFORE the optimizer is built, so it gets no gradient, no
    # optimizer slot and no weight decay. Applied on resume too, where the resumed state
    # tensors must then be bit-identical to the source.
    # training.freeze_state_from names the source RUN; its final checkpoint is read from
    # <out_root>/out/<run>/.
    if t.freeze_state_from:
        src_path = final_checkpoint_path(os.path.join(cfg.system.out_root, "out", t.freeze_state_from))
        src = torch.load(str(src_path), map_location="cpu", weights_only=False, mmap=True)
        print(model.freeze_state_tower(src["model"], verify_existing=checkpoint is not None)
              + f" | source={src_path}")
        del src

    block_size = model.config.block_size
    tokens_per_iter = t.global_batch_size * block_size
    checkpoint_manager.tokens_per_iter = tokens_per_iter  # locates the pre-decay save
    if master_process:
        print("Batch configuration:")
        print(f"  micro_batch_size: {t.micro_batch_size} (per GPU)")
        print(f"  global_batch_size: {t.global_batch_size}")
        print(f"  num_gpus: {ddp_world_size}")
        print(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")
        print(f"  tokens per iteration: {tokens_per_iter:,}")

    # Data
    data_dir = os.path.join(cfg.system.data_root, "data", cfg.data.dataset)
    train_bin_path = os.path.join(data_dir, "train.bin")
    val_bin_path = os.path.join(data_dir, "val.bin")
    for path in (train_bin_path, val_bin_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {os.path.basename(path)} at {path}")
    train_dataset = TokenDataset(train_bin_path, block_size)

    # A fresh DDP run starts rank r's stream at a random offset drawn from seed + r; a
    # 1-process run starts at 0. On resume, `legacy` gives every rank the offset saved by
    # rank 0 (how all paper runs resumed; docs/REPRODUCIBILITY.md), `per_rank` gives each
    # rank its own offset back, which continues the uninterrupted stream exactly.
    own_start_offset = (fresh_start_offset(base_seed, ddp_rank, len(train_dataset), t.sampler_max_start_offset)
                        if ddp else 0)
    sampler_start_offset = own_start_offset
    sampler_samples_seen_per_rank = 0
    if checkpoint is not None and 'sampler_offset' in checkpoint:
        saved_offset = int(checkpoint['sampler_offset'])
        if t.sampler_resume == "legacy":
            sampler_start_offset = saved_offset
        elif master_process and saved_offset != own_start_offset:
            raise ValueError(f"sampler_resume=per_rank: rank 0 offset {own_start_offset} does not match "
                             f"the checkpoint's {saved_offset} (different seed or world size?)")
        sampler_samples_seen_per_rank = int(checkpoint.get('sampler_samples_seen_per_rank', 0))
    print(f"Train sampler: rank {ddp_rank}/{ddp_world_size}, len={len(train_dataset):,}, "
          f"start_offset={sampler_start_offset:,}, samples_seen_per_rank={sampler_samples_seen_per_rank:,}")
    train_sampler = FixedRandomChunkDistributedSampler(
        dataset_len=len(train_dataset),
        num_replicas=ddp_world_size,
        rank=ddp_rank,
        block_size=block_size,
        chunk_size_units=t.chunk_size_units,
        seed=t.chunk_shuffle_seed,
        start_offset=sampler_start_offset,
        resume_samples_seen_per_rank=sampler_samples_seen_per_rank,
        balanced=t.sampler_fix,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=t.micro_batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=t.num_workers,
        pin_memory=(device_type == 'cuda'),
        drop_last=False,
    )
    print(f"Dataset: {len(train_dataset.data):,} tokens, "
          f"{t.max_tokens:,} to train (~{t.max_tokens // tokens_per_iter:,} iterations)")
    if master_process and sched.decay_lr:
        print(f"LR schedule: warmup to {sched.warmup_tokens:,} tokens, constant to {decay_start_tokens:,}, "
              f"linear decay to {t.max_tokens:,} (pre-decay checkpoint at ~{decay_start_tokens:,})")

    # Creating the iterator draws from the torch RNG, so it stays here, after model init and
    # before the resumed RNG state is restored.
    train_iter = iter(train_loader)

    def get_batch():
        nonlocal train_iter, sampler_samples_seen_per_rank
        try:
            x, y = next(train_iter)
        except StopIteration:
            if master_process:
                print(f"Dataset pass complete at iter {iter_num}. Creating new iterator...")
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        sampler_samples_seen_per_rank += x.shape[0]
        if device_type == 'cuda':
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    model.to(device)
    optimizer = model.configure_optimizers(cfg.optimizer.weight_decay, cfg.optimizer.learning_rate,
                                           (cfg.optimizer.beta1, cfg.optimizer.beta2), device_type)
    if checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as exc:
            print("Optimizer state mismatch; reinitializing optimizer state.")
            print(f"   Reason: {exc}")
        # RNG states must be CPU ByteTensors, but map_location may have moved them.
        if 'cpu_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['cpu_rng_state'].cpu().byte())
        if 'cuda_rng_state' in checkpoint and device_type == 'cuda':
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu().byte())
    checkpoint = None

    if cfg.system.compile:   # torch.compile(model), default mode: what every paper run used
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    @torch.no_grad()
    def estimate_loss():
        """Mean val loss over a fixed set of random val windows (same windows every call)."""
        model.eval()
        eval_batch_size = t.micro_batch_size * gradient_accumulation_steps
        eval_iters = max(1, math.ceil(t.eval_total_tokens / (eval_batch_size * block_size)))
        losses = torch.zeros(eval_iters)
        eval_stats_sums = {}
        eval_stats_occurrences = {}
        val_rng = torch.Generator().manual_seed(t.val_seed)
        data = np.memmap(val_bin_path, dtype=np.uint16, mode='r')
        for k in trange(eval_iters, disable=not master_process):
            ix = torch.randint(len(data) - block_size, (eval_batch_size,), generator=val_rng)
            X = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
            Y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
            if device_type == 'cuda':
                X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)
            else:
                X, Y = X.to(device), Y.to(device)
            with ctx:
                loss_total = 0.0
                stats_sums_iter = {}
                stats_occurrences_iter = {}
                for micro_step in range(gradient_accumulation_steps):
                    start = micro_step * t.micro_batch_size
                    end = start + t.micro_batch_size
                    _, loss_slice, stats = model(X[start:end], Y[start:end])
                    loss_total += float(loss_slice.detach())
                    if stats:
                        _accumulate_stats(stats, stats_sums_iter, stats_occurrences_iter)
            losses[k] = loss_total / max(max(stats_occurrences_iter.values(), default=0), 1)
            for key, total in stats_sums_iter.items():
                eval_stats_sums[key] = eval_stats_sums.get(key, 0.0) + total
                eval_stats_occurrences[key] = eval_stats_occurrences.get(key, 0) + stats_occurrences_iter.get(key, 0)
        out = {'val': losses.mean()}
        for key, value in _summarize_stats(eval_stats_sums, eval_stats_occurrences).items():
            out[f"val_{key}"] = value
        model.train()
        return out

    def run_eval(tag):
        """Evaluate, print one `<tag> | iter ...` line, and return (losses, wandb metrics)."""
        eval_start = time.perf_counter()
        losses = estimate_loss()
        eval_time = time.perf_counter() - eval_start
        print(f"{tag} | iter {iter_num:>6} | val nll: {losses.get('val_token_nll', float('nan')):.4f} "
              f"(ppl {losses.get('val_token_ppl', float('nan')):.2f}) | "
              f"tokens: {_format_tokens(iter_num * tokens_per_iter)} | eval: {eval_time*1000:.0f}ms")
        metrics = {"sys/eval_ms": eval_time * 1000.0}
        metrics.update({f"val/{key[4:]}": value for key, value in losses.items() if key.startswith("val_")})
        return losses, metrics

    def get_lr(it):
        """Linear warmup, constant, then linear decay to min_lr, all in tokens."""
        if it == 0:
            it = 1
        tokens_seen = it * tokens_per_iter
        if tokens_seen < sched.warmup_tokens:
            return cfg.optimizer.learning_rate * tokens_seen / sched.warmup_tokens
        if tokens_seen < decay_start_tokens:
            return cfg.optimizer.learning_rate
        if tokens_seen >= t.max_tokens:
            return sched.min_lr
        decay_ratio = (tokens_seen - decay_start_tokens) / sched.lr_decay_tokens
        coeff = 1.0 - decay_ratio
        return sched.min_lr + coeff * (cfg.optimizer.learning_rate - sched.min_lr)

    def save_checkpoint(**kind):
        checkpoint_manager.save_checkpoint(
            model_state=raw_model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            model_args=model_args,
            iter_num=iter_num,
            best_val_loss=best_val_loss,
            tokens_seen=iter_num * tokens_per_iter,
            config=OmegaConf.to_container(cfg, resolve=True),
            wandb_run_id=wandb_run_id,
            sampler_offset=int(sampler_start_offset),
            sampler_samples_seen_per_rank=int(sampler_samples_seen_per_rank),
            next_eval_tokens=next_eval_tokens,
            **kind,
        )

    def reduce_stats_for_logging(local_total_loss, local_stats_sums, local_stats_occurrences):
        """All-reduce the scalar logging accumulators across DDP ranks."""
        if not ddp:
            return (local_total_loss / gradient_accumulation_steps,
                    _summarize_stats(local_stats_sums, local_stats_occurrences))
        local_nll_sum = local_stats_sums.get("token_nll_sum", float("nan"))
        local_nll_count = local_stats_sums.get("token_nll_count", 0.0)
        local_nll = local_nll_sum / max(local_nll_count, 1.0)
        if local_nll < t.anomaly_nll_threshold or not math.isfinite(local_nll):
            print(f"PRE-REDUCE rank={ddp_rank} | local_loss={local_total_loss:.6f} | "
                  f"nll_sum={local_nll_sum:.6f} | nll_count={local_nll_count:.0f} | nll={local_nll:.6f}",
                  flush=True)
        loss_tensor = torch.tensor([local_total_loss], device=torch.device(device), dtype=torch.float64)
        all_reduce(loss_tensor, op=ReduceOp.SUM)
        global_avg_loss = (loss_tensor.item() / float(ddp_world_size)) / gradient_accumulation_steps
        all_keys = sorted(set(local_stats_sums) | set(local_stats_occurrences))
        if not all_keys:
            return global_avg_loss, {}
        packed = torch.zeros((len(all_keys), 2), device=torch.device(device), dtype=torch.float64)
        for i, key in enumerate(all_keys):
            packed[i, 0] = float(local_stats_sums.get(key, 0.0))
            packed[i, 1] = float(local_stats_occurrences.get(key, 0))
        all_reduce(packed, op=ReduceOp.SUM)
        global_stats_sums = {key: float(packed[i, 0].item()) for i, key in enumerate(all_keys)}
        global_stats_occurrences = {key: int(round(packed[i, 1].item())) for i, key in enumerate(all_keys)}
        return global_avg_loss, _summarize_stats(global_stats_sums, global_stats_occurrences)

    if cfg.logging.wandb_log and master_process:
        import wandb
        wandb_dir = prepare_wandb_dir_from_config(cfg)
        run = wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            entity=cfg.logging.wandb_entity,
            tags=cfg.experiment.get('tags', []) if 'experiment' in cfg else [],
            id=wandb_run_id,
            resume="allow",
            **({"dir": wandb_dir} if wandb_dir is not None else {}),
        )
        wandb_run_id = run.id
    elif wandb_run_id is None:
        wandb_run_id = f"local_{int(time.time())}"  # checkpoints always carry a run id

    t0 = time.time()
    val_metrics = None  # logged together with the next train line
    tokens_seen_start = iter_num * tokens_per_iter
    if next_eval_tokens is None:
        skip = t.skip_first_eval and not t.eval_only
        next_eval_tokens = tokens_seen_start + (t.eval_interval_tokens if skip else 0)
    model.train()
    print(f"Starting training loop at iter_num={iter_num}, tokens_seen={tokens_seen_start:,}")

    while True:
        lr = get_lr(iter_num) if sched.decay_lr else cfg.optimizer.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        tokens_seen = iter_num * tokens_per_iter
        if tokens_seen >= next_eval_tokens and master_process:
            losses, metrics = run_eval("eval")
            while next_eval_tokens <= tokens_seen:
                next_eval_tokens += t.eval_interval_tokens
            if cfg.logging.wandb_log:
                val_metrics = metrics
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']

        # Named snapshots every save_every tokens (plus the pre-decay one), and the rolling
        # resume checkpoint ckpt.pt every rolling_save_every tokens before the decay.
        should_save, is_pre_decay = checkpoint_manager.should_save(tokens_seen, iter_num)
        if should_save:
            save_checkpoint(is_pre_decay=is_pre_decay)
        if checkpoint_manager.should_save_rolling(tokens_seen, iter_num):
            save_checkpoint(is_rolling=True)

        if iter_num == 0 and t.eval_only:
            break

        # Forward, backward, update
        total_loss_tensor = torch.zeros(1, device=device)
        train_stats_sums = {}
        train_stats_occurrences = {}
        for micro_step in range(gradient_accumulation_steps):
            X, Y = get_batch()
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                _, loss, stats = model(X, Y)
                total_loss_tensor += loss.detach()
                loss = loss / gradient_accumulation_steps
            if stats is not None:
                _accumulate_stats(stats, train_stats_sums, train_stats_occurrences)
            loss.backward()
        total_loss = total_loss_tensor.item()  # single GPU sync after all micro-steps

        # Per-rank anomaly diagnostics (print only).
        local_nll_sum = train_stats_sums.get("token_nll_sum", float("nan"))
        local_nll_count = train_stats_sums.get("token_nll_count", 0.0)
        local_nll = local_nll_sum / max(local_nll_count, 1.0)
        loss_finite = torch.isfinite(total_loss_tensor).all().item()
        is_anomalous = local_nll < t.anomaly_nll_threshold or not loss_finite or not math.isfinite(local_nll)
        if is_anomalous:
            print(f"ANOMALY rank={ddp_rank} iter={iter_num} | local_total_loss={total_loss:.6f} | "
                  f"local_nll_sum={local_nll_sum:.6f} | local_nll_count={local_nll_count:.0f} | "
                  f"local_nll={local_nll:.6f} | loss_finite={loss_finite}", flush=True)

        if cfg.optimizer.grad_clip != 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
            if is_anomalous or not math.isfinite(grad_norm.item()):
                print(f"GRAD rank={ddp_rank} iter={iter_num} | grad_norm={grad_norm.item():.6f} | "
                      f"finite={torch.isfinite(grad_norm).item()}", flush=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if is_anomalous and device_type == 'cuda':
            torch.cuda.synchronize()
            print(f"POST-OPTIM rank={ddp_rank} iter={iter_num} | CUDA sync OK", flush=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        avg_loss, train_metrics = reduce_stats_for_logging(total_loss, train_stats_sums, train_stats_occurrences)
        if iter_num % t.log_interval == 0 and master_process:
            avg_nll_loss = train_metrics.get("token_nll")
            train_ppl = train_metrics.get("token_ppl", float("nan"))
            tok_per_s = tokens_per_iter / dt if dt > 0 else float('nan')
            peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3) if device_type == 'cuda' else 0.0
            nll_str = f" | nll: {avg_nll_loss:.4f}" if avg_nll_loss is not None else ""
            print(f"train | iter {iter_num:>6} | total: {avg_loss:.4f}{nll_str} | ppl {train_ppl:>7.2f} | "
                  f"lr: {lr:.2e} | {dt*1000:.0f}ms | tok/s: {tok_per_s:,.0f} | "
                  f"peak: {peak_gib:.2f}GiB | tokens: {tokens_seen:,}")
            if cfg.logging.wandb_log:
                log_dict = {
                    "iter": iter_num,
                    "tokens_seen": tokens_seen,
                    "train/total_loss": avg_loss,
                    "train/perplexity": train_ppl,
                    "lr": lr,
                }
                log_dict.update({f"train/{key}": value for key, value in train_metrics.items()})
                if val_metrics is not None:
                    log_dict.update(val_metrics)
                    val_metrics = None
                wandb.log(log_dict)

        iter_num += 1

        tokens_seen = iter_num * tokens_per_iter
        if tokens_seen >= t.max_tokens:
            if master_process:
                print(f"Reached max_tokens: {tokens_seen:,} >= {t.max_tokens:,}")
                print("Running final evaluation...")
                _, metrics = run_eval("final")
                if cfg.logging.wandb_log:
                    wandb.log({"iter": iter_num, "tokens_seen": tokens_seen, **metrics})
                save_checkpoint(is_final=True)
            break

    # Coordinated shutdown. The final eval and checkpoint above run on rank 0 only; without
    # this barrier the other ranks exit early, their torchelastic agents wait in the exit
    # barrier until its 300 s timeout, and the allocation idles to the wall clock. Changes
    # no number.
    if ddp:
        torch.distributed.barrier(device_ids=[ddp_local_rank])
        destroy_process_group()
    if master_process and cfg.logging.wandb_log:
        wandb.finish()
    # Leave immediately: interpreter teardown (compile / NCCL / wandb atexit handlers) is
    # where per-rank skew used to creep back in.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
