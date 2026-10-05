# Allowed resolved-config differences (G2, and G3's model_args)

G2 compares every paper config's resolved Hydra YAML with `gates/golden/g2/`, resolved from
the configs the paper runs were trained with. A difference fails the gate unless it is
listed in the block below, with a reason. Only two kinds exist:

- `drop [<config-glob>] <key>`: a dead key (never read by any code path the paper uses) may
  disappear from the matching configs (every config when no glob is given). A dotted prefix
  drops the whole subtree.
- `set <config-glob> <key> <yaml-value>`: the key must have exactly this value in the
  matching configs. Use it for explicit pins (e.g. the attention backend that actually ran)
  and for keys that replace a dropped one.

Keys under `model.config.` also apply to G3's `model_args` (the dataclass that is saved in
every checkpoint).

```allowed
# Keys no code reads, and removed embedding-probe hooks
drop training.val_max_tokens
drop training.always_save_checkpoint
drop training.two_tower_emb_probe_init
drop training.emb_snapshot_every
# Former literals in train.py, now config keys with the same values
set * training.world_size 8
set * training.val_seed 42
set * training.anomaly_nll_threshold 0.5
set * training.sampler_max_start_offset 1000000
set * training.sampler_resume legacy
set * training.num_workers 0
# The corpus recipe, formerly constants and environment variables in prepare.py
set * data.prepare.hf_repo HuggingFaceFW/fineweb-edu
set * data.prepare.hf_revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
set * data.prepare.sample 100BT
set * data.prepare.max_files 36
set * data.prepare.file_select head
set * data.prepare.file_offset 0
set * data.prepare.shuffle_seed 2357
set * data.prepare.shuffle_buffer_size 200000
set * data.prepare.val_fraction 0.0005
set * data.prepare.val_split_seed 42
set * data.prepare.batch_size 2000
set * data.prepare.tokenizer gpt2
set * data.prepare.num_proc 86

# Attention backend pinned to what ran. `auto` resolves to FlashAttention when a flash-attn
# wheel is importable, else flex; the training venv had no flash-attn wheel, so every
# two-tower run used flex. Same kernel choice here, so no numeric change.
set s_two_tower_* model.config.attn_backend flex

# Freeze-and-retrain: the config names the source RUN, and train.py loads that run's one
# ckpt_tokens_*_final.pt from <system.out_root>/out/<run>/ (training.checkpointing.
# final_checkpoint_path) instead of an absolute checkpoint path. Same file as before
# (ckpt_tokens_20000145408_final.pt of these two runs).
set s_two_tower_frz_seq6src_* training.freeze_state_from s_two_tower_seq6_20b
set s_two_tower_frz_tt6src_* training.freeze_state_from s_two_tower_w0_equal6_20b

# Opt-in train-loop speed flags that no paper config turned on (all were false, so every
# paper run took the system.compile path, which stays). Their code is removed.
drop training.compile
drop training.compile_scope
drop training.compile_mode
drop training.compile_dynamic
drop training.fused_adamw
drop training.tf32
drop training.chunked_ce
drop training.chunked_ce_rows

# experiment.tags (W&B metadata only): 20B two-tower runs were tagged `2b`, and the 6+6
# and shared parallel arms were tagged `seq`. One scheme now: scale, family, variant,
# 20b, fw100. Seed replicates inherit their base's tags.
set s_full_attention_20b_fw100 experiment.tags [s, full_attention, 20b, fw100]
set s_full_attention_20b_fw100_untied experiment.tags [s, full_attention, untied, 20b, fw100]
set s_sps_w64_20b_fw100 experiment.tags [s, sps, w64, 20b, fw100]
set s_sps_w64_20b_fw100_untied experiment.tags [s, sps, w64, untied, 20b, fw100]
set s_two_tower_afsps_faithful_20b experiment.tags [s, two_tower, afsps, shared, 20b, fw100]
set s_two_tower_w0_equal_20b experiment.tags [s, two_tower, parallel, 20b, fw100]
set s_two_tower_w0_equal_tiedattn_20b experiment.tags [s, two_tower, parallel, tied_attn, 20b, fw100]
set s_two_tower_w0_s1152p3456_20b experiment.tags [s, two_tower, parallel, mlp_realloc, 20b, fw100]
set s_two_tower_w0_equal6_20b experiment.tags [s, two_tower, parallel, 20b, fw100]
set s_two_tower_asym_20b experiment.tags [s, two_tower, asym, 20b, fw100]
set s_two_tower_seq12_20b experiment.tags [s, two_tower, sequential, 20b, fw100]
set s_two_tower_seq6_20b experiment.tags [s, two_tower, sequential, 20b, fw100]
set s_two_tower_seq6_slim_20b experiment.tags [s, two_tower, sequential, slim, 20b, fw100]
set s_two_tower_seq6_cpause_20b experiment.tags [s, two_tower, sequential, pause_input, 20b, fw100]
set s_two_tower_seq3p9_20b experiment.tags [s, two_tower, sequential, 20b, fw100]
set s_two_tower_seq9p3_20b experiment.tags [s, two_tower, sequential, 20b, fw100]
set s_two_tower_w0_shared_20b experiment.tags [s, two_tower, shared, 20b, fw100]
set s_two_tower_seq12_tied_20b experiment.tags [s, two_tower, sequential, shared, 20b, fw100]
set s_two_tower_frz_seq6src_early_20b experiment.tags [s, two_tower, freeze_retrain, 20b, fw100]
set s_two_tower_frz_seq6src_final_20b experiment.tags [s, two_tower, freeze_retrain, 20b, fw100]
set s_two_tower_frz_seq6src_post_20b experiment.tags [s, two_tower, freeze_retrain, 20b, fw100]
set s_two_tower_frz_tt6src_early_20b experiment.tags [s, two_tower, freeze_retrain, 20b, fw100]
set s_two_tower_frz_tt6src_final_20b experiment.tags [s, two_tower, freeze_retrain, 20b, fw100]
set s_two_tower_frz_tt6src_post_20b experiment.tags [s, two_tower, freeze_retrain, 20b, fw100]
set s_full_attention_20b_fw100_untied_seed2 experiment.tags [s, full_attention, untied, 20b, fw100]
set s_sps_w64_20b_fw100_seed2 experiment.tags [s, sps, w64, 20b, fw100]
set s_sps_w64_20b_fw100_seed3 experiment.tags [s, sps, w64, 20b, fw100]
set s_sps_w64_20b_fw100_untied_seed2 experiment.tags [s, sps, w64, untied, 20b, fw100]
set s_two_tower_frz_seq6src_early_20b_seed2 experiment.tags [s, two_tower, freeze_retrain, 20b, fw100]
set s_two_tower_frz_seq6src_final_20b_seed2 experiment.tags [s, two_tower, freeze_retrain, 20b, fw100]
set s_two_tower_frz_seq6src_post_20b_seed2 experiment.tags [s, two_tower, freeze_retrain, 20b, fw100]
set s_two_tower_seq12_20b_seed2 experiment.tags [s, two_tower, sequential, 20b, fw100]
set s_two_tower_seq12_20b_seed3 experiment.tags [s, two_tower, sequential, 20b, fw100]
set s_two_tower_seq12_tied_20b_seed2 experiment.tags [s, two_tower, sequential, shared, 20b, fw100]
set s_two_tower_seq6_20b_seed2 experiment.tags [s, two_tower, sequential, 20b, fw100]
set s_two_tower_seq6_slim_20b_seed2 experiment.tags [s, two_tower, sequential, slim, 20b, fw100]
set s_two_tower_w0_equal6_20b_seed2 experiment.tags [s, two_tower, parallel, 20b, fw100]
set s_two_tower_w0_equal_20b_seed2 experiment.tags [s, two_tower, parallel, 20b, fw100]
set s_two_tower_w0_equal_20b_seed3 experiment.tags [s, two_tower, parallel, 20b, fw100]

# Dead code. The two-tower pred tower never had keys of its own: every run
# used pred_window = state_pred_window = 0 and pred_self_module = fused, the only values
# the model accepted, so the fields are gone (older checkpoints' model_args still load:
# modeling.models.model.config_args drops them). SPS likewise only ever took
# predict_embedding = constant. The `eval` config group and training.sampler_type
# (a single-valued switch) were read by no code.
drop s_two_tower_* model.config.pred_window
drop s_two_tower_* model.config.state_pred_window
drop s_two_tower_* model.config.pred_self_module
drop s_sps_* model.config.predict_embedding
drop eval
drop training.sampler_type
```
