# MIST HER2 正式训练锁定单（PPO velocity limit 1.0）

本锁定单用于基于 `9eda1d4` 代码线启动一次新的 MIST HER2 正式训练。它把已选训练目标和关键超参数写进独立配置，避免正式 Flow/PPO 阶段因 CLI 漏参而使用不同的速度修正尺度。

## 锁定设置

| 项目 | 正式值 |
|---|---|
| 官方数据 | MIST HER2 官方 train 4,642 对；官方 test 1,000 对 |
| 训练缓存 | 只由官方 train 4,642 对生成；校验原图字节数和 SHA-256 |
| 训练输入 | 配对共享坐标随机裁块 512×512，batch 1 |
| 正式评测 | 官方 test 原生 1024×1024 整图 |
| seed / 精度 | 1234 / `auto`（日志必须记录实际 BF16 或 FP32） |
| Flow | 40 epochs；64 px 四相位区域 CFM |
| 生成器目标 | 区域 CFM + 0.1 PatchNCE + 0.1 对抗项 |
| Flow 路径 | 4 步 Euler；策略动作在零基第 2 步 |
| PPO | 10 epochs；每状态 4 候选；128 px 四相位区域奖励；`min_reward_spread=1e-5` |
| PPO 速度上限 | `velocity_limit=1.0`，配置文件和 PPO CLI 双重固定 |

完整参数位于 [mist_her2_formal_ppo1.json](../../configs/mist_her2_formal_ppo1.json)。该值 `1.0` 是在前一轮 pilot-dev 尺度扫描后选定的；对应 pilot 结果属于探索性结果。现在把参数冻结，后续不再在同一 pilot-dev 上继续扫描。最终效能结论只能由正式训练完成后、按锁定协议运行的官方 test 支持。官方 test 不参与阈值、参数或 checkpoint 选择。

## 启动命令

从仓库根目录执行。使用新的缓存和运行目录；不要从 Flow1000、PPO1000 或旧的 shortcut checkpoint 续训。

```bash
python -m morphoflow_ihc prepare \
  --manifest datasets/MIST_HER2/manifest.csv \
  --config configs/mist_her2_formal_ppo1.json \
  --output data/cache_mist_her2_formal_ppo1

python -m morphoflow_ihc train \
  --cache data/cache_mist_her2_formal_ppo1 \
  --config configs/mist_her2_formal_ppo1.json \
  --output runs/formal_mist_her2_ppo1_seed1234/flow.pt \
  --device cuda --seed 1234 --batch 1 --amp auto \
  --epochs 40 --no-time-limit

python -m morphoflow_ihc train-rl \
  --cache data/cache_mist_her2_formal_ppo1 \
  --checkpoint runs/formal_mist_her2_ppo1_seed1234/flow.pt \
  --output runs/formal_mist_her2_ppo1_seed1234/joint.pt \
  --device cuda --seed 1234 --batch 1 --amp auto \
  --policy-velocity-limit 1.0 --epochs 10 --no-time-limit
```

启动前核对缓存索引为 4,642 个 train pair、无官方 test 图进入 cache entries；确认 Flow checkpoint 记录的配置与锁定 JSON 完全一致，PPO 日志含 `policy_velocity_limit: 1.0`。Flow 和 PPO 要共用相同 cache index、seed、batch 和有效 AMP 精度。使用此前 pilot 的 4,514 对 cache 或已有输出目录都不符合本锁定单。

## 运行中与结束后记录

保存本次 commit、工作区状态、锁定配置 SHA-256、官方 manifest/cache index SHA-256、GPU/驱动/PyTorch/CUDA、实际 AMP 精度、每阶段更新数、每轮损失与 PPO 的 spread/signal fraction/accepted/KL、峰值显存和 wall-clock。保留 checkpoint SHA-256；图像/权重不上传 Git，除非另行决定。

训练完成并冻结 checkpoint 后，才对官方 1,000 对 test 运行一次预先固定的 1024×1024 全图评测。以连续区域 support MAE 和 DAB 量误差为主要区域指标；本次 pilot 的 support F1 在所有模型上相同，1% 窗口阈值无区分度，不能再把它当主要证据。可记录它作为诊断项并同时报告预测阳性窗口比例。报告 DAB 过量/欠量、输出裁剪/饱和与固定示例，避免只报平均分。

## 结论边界

本轮 pilot 在 128 对开发样本上显示了 PPO 正向方向，但 velocity scale 经同一开发集选择，且总量误差仍高。因此这份配置是“冻结后进入正式训练”的方案，不等同于 PPO 已被确认有效。正式 test 未评估前，不引用 pilot 区间为最终效能结论；患者级独立性也未由文件名前缀证明。
