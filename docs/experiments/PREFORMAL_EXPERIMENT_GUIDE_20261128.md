# MorphoFlow-IHC 正式实验前预实验指导

适用代码：`fix/flow-design-audit-20261128` 的 `273f2f8` 及其后续提交。目标是在正式 40+10 轮训练前排除数据、梯度、混合精度、检查点和推理链路错误；短跑只证明实现可运行，**不证明模块改善了病理结果**。

## 当前已经完成的检查

提交 [`273f2f8`](https://github.com/zhengboyu156-create/MorphoFlow-IHC/commit/273f2f89b50eb09a18524edd484e7def94860b4b) 的审计报告已记录：28 项测试通过；RTX 5070 Ti 上真实 MIST 512 px 裁块的 BF16/FP16 Flow 更新、1024 px 全图推理、单图 Fiji 测量均通过。若本地使用的就是该提交、相同依赖和目标 GPU，不必重复大批相同 smoke test。

短测里 PPO 候选奖励差曾低于 `min_reward_spread=1e-5`，所以更新按设计跳过。冷启动时这不代表程序故障，也不证明 PPO 有效；不要为了让它更新而临时降低阈值。

## 本实验必须锁定的配置

| 项目 | MIST HER2 正式设置 |
|---|---|
| 数据 | 官方 4,642 对训练图；1,000 对 `valA/valB` 是官方测试队列，不是单独验证集 |
| 输入 | 训练随机配对裁块 512×512，batch 1；测试原生 1024×1024 整图 |
| 随机种子 | 1234 |
| 精度 | `auto`（记录实际解析到的精度，如 BF16） |
| Flow | 40 个数据轮次；区域速度尺度 64 px；推理路径 4 步 Euler |
| 生成器损失 | 区域 CFM + `0.1 × PatchNCE` + `0.1 × adversarial` |
| PPO | 10 个数据轮次；动作在第 2 步；每状态 4 个候选；低信号门槛 `1e-5` |
| 奖励尺度 | `reward_region=128 px`；四个网格相位分别计分后取平均 |
| 检查点 | 新训练用格式 10；旧 Flow40 权重不续训本方法 |

数据说明没有证实患者级独立性；`source_prefix` 不能自动视为患者 ID。论文和报告不要写成患者独立验证。

## 预实验步骤与放行标准

### 1. 固定代码、环境和输出目录

```bash
git rev-parse HEAD
git status --short
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

记录提交号、GPU、驱动、PyTorch/CUDA 版本、配置与 manifest 的 SHA-256。正式实验使用新输出路径，不能覆盖历史 checkpoint 或短跑结果。

放行条件：代码含格式 10 的训练/续训约束，工作区干净，CUDA 和目标 GPU 与预期一致。

### 2. 重新生成并核对官方训练缓存

官方清单的 `trainA/trainB` 是训练集，`valA/valB` 是官方测试集。新格式 10 正式实验应重新 `prepare`，以在缓存时核对官方训练源文件的字节数和 SHA-256。

```bash
python -m morphoflow_ihc prepare \
  --manifest datasets/MIST_HER2/manifest.csv \
  --config configs/mist_her2.json \
  --output data/cache_mist_format10
```

核对 `data/cache_mist_format10/index.json`：`split_counts.train` 为 4,642、`split_counts.test` 为 1,000；训练 `entries` 仅含训练对；每个 pair 恰有一张 H&E 和一张 IHC；MPP 与配置一致；训练输入裁块为 512 px。缓存建立失败、数量不符或哈希不匹配时停止，不要继续用旧缓存。

### 3. 确认现有自动化测试覆盖

```bash
python -m pytest -q
```

`273f2f8` 的记录为 **28 项通过**。若代码未变化且本机环境与报告一致，可引用这次结果；代码、依赖或 CUDA 环境变更后应重新跑。测试通过只说明被覆盖的行为正确，不等于整套损失已经证明有收益。

重点确认这些梯度边界：

| 子模块 | 正确的梯度/状态行为 |
|---|---|
| 区域 CFM | 有限损失，速度网络收到有限非零梯度 |
| PatchNCE | 使用推理同款四步生成样本；梯度能从生成图回到速度网络和查询侧形态编码/投影模块；H&E 键侧按设计停止梯度 |
| 对抗项 | critic 的 real/fake 损失更新 critic；生成器对抗损失经冻结的 critic 回传到速度网络，不能更新 critic 参数或运行统计 |
| PPO | 候选奖励有限且有界；低于门槛的组被跳过；有效动作概率以 FP32 计算；策略更新不应误更新策略之外的参数 |

### 4. 用新缓存做一次短程真实数据烟雾测试

只有在第 2 步生成了新缓存，或本机没有与审计报告相同的运行环境时，才需要额外跑此项。短跑使用训练数据，不能用官方测试集选参数。

```bash
python -m morphoflow_ihc train \
  --cache data/cache_mist_format10 --config configs/mist_her2.json \
  --output runs/preflight_20261128/flow.pt --device cuda \
  --seed 1234 --batch 1 --amp auto --steps 8 --save-every 8 --no-time-limit

python -m morphoflow_ihc train-rl \
  --cache data/cache_mist_format10 --checkpoint runs/preflight_20261128/flow.pt \
  --output runs/preflight_20261128/policy.pt --device cuda \
  --seed 1234 --batch 1 --amp auto --groups 8 --save-every 1 --no-time-limit
```

检查 Flow 的 `flow_loss`、`nce_loss`、`critic_loss`、`adversarial_loss` 均为有限数；checkpoint 可重新加载；PPO 的 reward/spread/KL 有限，并能看到低信号组按门槛跳过。只有 8 步 Flow 的冷启动可能令 8 个 PPO 组全被跳过；这不应阻止先开始正式 Flow，也不能作为“PPO 有效”的证据。不要降低信号门槛来满足预设结果。

短测只验证运行链路。它不要求短程 PSNR/FID 提升，也不应因短程输出不好就反复挑 seed 或 checkpoint。

### 5. 核对两阶段训练参数完全一致

当前从 Flow checkpoint 开始联合训练时，代码强制检查训练缓存，但尚未强制比较阶段切换时的 seed、batch 和有效精度。因此两个命令都要显式使用同一组 `--seed 1234 --batch 1 --amp auto`。不要只在某一阶段改 batch 或 AMP。

Flow 正式命令：

```bash
python -m morphoflow_ihc train \
  --cache data/cache_mist_format10 --config configs/mist_her2.json \
  --output runs/formal_20261128/flow.pt --device cuda \
  --seed 1234 --batch 1 --amp auto --epochs 40 --no-time-limit
```

PPO 联合阶段从上一步新产生的格式 10 checkpoint 开始：

```bash
python -m morphoflow_ihc train-rl \
  --cache data/cache_mist_format10 --checkpoint runs/formal_20261128/flow.pt \
  --output runs/formal_20261128/joint.pt --device cuda \
  --seed 1234 --batch 1 --amp auto --epochs 10 --no-time-limit
```

两阶段都使用同一官方训练缓存和配置；Flow 从头开始，不能接着旧 shortcut 目标训练的 Flow40 checkpoint 跑。

## 模块“有效”的判定边界

预实验能确认损失项接入正确、梯度能到达预期模块、奖励在给定样本上可计算、训练和推理能跑通。它**不能**确认模块提升最终图像质量。

对效果的验证放在正式结果分析中：固定同一训练划分、更新计划和 seed；比较预先定义的基线/消融；策略推理贡献用同一个联合 checkpoint 分别开启和 `--disable-policy`，不能换生成器权重来做这个对照。最终 MIST 指标只在训练选择固定后对完整 1,000 对官方测试图运行；不要在预实验阶段反复用这 1,000 对调阈值、权重或挑 checkpoint。

## 最终放行清单

- [ ] 当前代码版本包含格式 10 和本指南所对应实现；记录 commit/config/manifest 哈希。
- [ ] 含 manifest SHA-256 的新训练缓存建立成功，4,642 训练对和 1,000 官方测试对数量正确，官方训练图哈希通过。
- [ ] 单元测试通过；真实训练短跑没有 NaN/Inf，loss 和梯度路径符合上表。
- [ ] 1024×1024 全图推理输出尺寸、像素有限；若本地环境和审计报告一致，引用已有测试，不重复跑全评测。
- [ ] Flow 与 PPO 两阶段显式使用同一 seed、batch 和 AMP；正式输出路径全新。
- [ ] 记录尚未解决的局限：患者独立性未证实、PPO 冷启动可跳过更新、短跑不证明模块有效。
- [ ] 完成以上项目后可开始正式 40+10；效果结论等待完整训练及锁定后的全测试集评测。
