# 可选 DeepSpeed ZeRO-2

日期: 2026-09-22

默认训练路径不变。不设 `USE_DEEPSPEED` 时，启动仍是 Accelerate 的 `--multi_gpu`，存盘仍走原来的 `save_checkpoint`。只有 `USE_DEEPSPEED=1` 才切到 ZeRO stage 2。

ZeRO-2 把优化器状态和梯度切到 8 张卡上，参数和激活仍留在每张卡。可训练参数大约 3B（Qwen3.5-2B + action expert + keypoint expert）时，每卡大约省 25–35GB 的优化器状态和梯度。它不减少前向激活，解决不了 Wan video loss 把显存打满的那类问题。

## 用法

环境里还没有 `deepspeed` 包。打开开关之前先装到 internvla 环境，不要装进 `/opt/conda`：

```bash
/home/a26160/ENV/internvla_a1_5/bin/python -m pip install deepspeed
```

```bash
# 原来的 DDP，默认
bash launch/internvla_a15_finetune_libero_plus_geop.sh

# ZeRO-2
USE_DEEPSPEED=1 bash launch/internvla_a15_finetune_libero_plus_geop.sh
```

日志里会打印 `USE_DEEPSPEED=0` 或 `USE_DEEPSPEED=1`。

## 改了哪些文件

### `launch/ds_zero2.json`（新文件）

- `bf16.enabled = true`
- `zero_optimization.stage = 2`
- `train_micro_batch_size_per_gpu`、`gradient_accumulation_steps`、`train_batch_size` 都是 `"auto"`，由 Accelerate 按启动时的进程数和 `batch_size` 填
- `gradient_clipping = 1.0`

没有写 `optimizer` 和 `scheduler`。优化器仍是训练脚本里的 AdamW，学习率仍由训练循环每个 batch 走一次。DeepSpeed 配置里再写一份会和代码里的优化器冲突。

`gradient_clipping` 必须写在这里。DeepSpeed 下 `Accelerator.clip_grad_norm_` 只汇报梯度范数，真正的裁剪发生在 DeepSpeed engine 里。1.0 与现有 AdamW preset 的 `grad_clip_norm` 一致。

### `launch/internvla_a15_finetune_libero_plus_geop.sh`

`USE_DEEPSPEED` 默认 `0`。

等于 `1` 时，`accelerate launch` 改为：

- `--use_deepspeed`
- `--deepspeed_config_file=launch/ds_zero2.json`

这时不再加 `--multi_gpu`。Accelerate 不允许 `--multi_gpu` 和 `--use_deepspeed` 同时出现。GPU 数量仍由 `--num_processes` 决定，进程由 DeepSpeed launcher 拉起。

等于 `0` 时，多卡仍是原来的 `--multi_gpu`。

### `src/lerobot/scripts/lerobot_train.py`

只在 `accelerator.distributed_type == DistributedType.DEEPSPEED` 时走新分支。

**梯度范数。** DeepSpeed 的 `clip_grad_norm_` 在 engine 还没有全局范数时返回 `None`。这时记成 0，避免 `grad_norm.item()` 在第一步崩掉。DDP 路径的返回值不是 `None`，行为不变。

**存 checkpoint。** 所有 rank 都要调用 `accelerator.save_state()`，因为 ZeRO 的优化器汇总是集合通信，不能只放在主进程里。写到：

```text
checkpoints/<step>/training_state/deepspeed/
```

主进程另外写原来的推理权重，供 `from_pretrained` 使用：

```text
checkpoints/<step>/pretrained_model/model.safetensors
checkpoints/<step>/pretrained_model/config.json
checkpoints/<step>/pretrained_model/train_config.json
checkpoints/<step>/pretrained_model/stats.json
checkpoints/<step>/training_state/training_step.json
```

这一支不调用 `optimizer.state_dict()`。ZeRO-2 下主进程拿到的是本卡的 1/8 分片，写出来的 `optimizer_state.safetensors` 不能用来恢复。

DDP 分支仍是原来的 `save_checkpoint`：主进程写 `pretrained_model/` 和完整的 `training_state/`（含 `optimizer_state.safetensors` 与 scheduler）。

**恢复。** DDP 的 `--resume` 仍在 `prepare()` 之前调用 `load_training_state`。DeepSpeed 的优化器状态要在 `prepare()` 之后用 `accelerator.load_state()` 从 `training_state/deepspeed` 读，步数从 `training_step.json` 读。用 DDP checkpoint 去 `USE_DEEPSPEED=1` 恢复会直接报错，因为那个目录不存在。

### `src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py`

DeepSpeed 打开 `bf16` 后会把全部浮点参数收成 bf16。训练前向里有几处把激活写成 fp32 再送进线性层，`F.linear` 报 `mat1 and mat2 must have the same dtype, but got Float and BFloat16`。第一次崩在 `_compute_video_loss` 的 `learnable_to_wan_proj`。

这三处改为先把输入转成该层 `weight.dtype`，线性层输出再 `.float()` 算 loss：

- `learnable_to_wan_proj`（`_compute_video_loss`）
- `action_out_proj`（action loss）
- `keypoint_out_proj`（当前帧和 future 关键点）

权重仍是 fp32 时输入也是 fp32，DDP 的数值路径不变。推理里的 `generate_video` 本来就会按 `learnable_to_wan_proj.weight.dtype` 对齐，这次只补了训练路径。

ZeRO-2 参数仍完整留在每张卡上，`prepare()` 之前的 `.to(cuda)` 没有改。

## 没有改的部分

- `src/lerobot/utils/train_utils.py` 里的 `save_checkpoint` / `load_training_state`。DDP 继续用它们。
- `Accelerator(...)` 的构造。`accelerate launch --use_deepspeed` 会让这个对象自动带上 DeepSpeed。`DistributedDataParallelKwargs(find_unused_parameters=True)` 只对 DDP 有效，DeepSpeed 下会被忽略。

## 权重怎么用

| 目的 | 用哪份 |
| --- | --- |
| 评测、或作为下一次的 `pretrained_path` | `pretrained_model/`（两种启动方式都有） |
| 中断后接着训，优化器也要恢复 | 必须和存盘时同一种启动方式。DDP 用 `training_state/optimizer_state.safetensors`，DeepSpeed 用 `training_state/deepspeed/` |
