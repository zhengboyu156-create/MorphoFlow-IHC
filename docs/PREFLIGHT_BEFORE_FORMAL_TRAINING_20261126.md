# 正式训练前预实验与放行清单

本清单用于本地适配 MorphoFlow-IHC。目的不是用短跑证明方法有效，而是在占用正式训练资源前确认：数据配对与划分可审计、三项生成器损失有有效梯度、策略奖励方向合理、策略确实影响部署轨迹，并且训练与推理能完整运行。

## 放行原则

按下列顺序执行；前一关未通过，不进入后一关。短程结果只能排除实现错误和明显不稳定，不能代替正式验证集、独立患者测试或病理结论。

| 关卡 | 需要回答的问题 | 放行条件 |
|---|---|---|
| 1. 环境与代码 | 当前提交、依赖、CUDA 是否可复现？ | 固定提交和环境记录；全部自动化测试通过；CUDA 可用时 GPU 前向、反向和检查点读写通过。 |
| 2. 数据 | 配对、物理尺度、划分和组织来源是否可靠？ | 配对清单无缺项/重复；H&E 与 IHC 尺寸和像素间距可核；训练/验证/测试按患者或最高层级来源隔离；完成配对错位审查。 |
| 3. 损失与梯度 | CFM、PatchNCE、GAN 是否都在训练预期模块？ | 损失有限；生成器参数有有限非零梯度；PatchNCE 到形态编码器和速度场、GAN 到速度场的梯度连通；critic 单独更新。 |
| 4. 策略与奖励 | PPO 是否能区分候选动作，奖励是否偏好合理染色？ | 合成受控例满足预期排序；真实验证 ROI 上奖励与预先指定的区域 DAB 误差方向一致；策略有被接受的有限更新且能改变输出。 |
| 5. 短程训练与推理 | 训练、续存、推理能否端到端工作？ | 短程日志无 NaN/Inf；流与策略检查点可重载；固定 seed 推理稳定，输出尺寸正确且无明显接缝；记录显存和吞吐。 |
| 6. 正式实验协议 | 比较是否公平、没有用测试集调参？ | 数据清单、配置、seed、训练步数和评价脚本冻结；所有消融从同一基线检查点/划分启动，只改预先指定因素；测试集只在最终评估时使用。 |

## 1. 固定环境与版本

在仓库根目录执行，并把输出保存到实验归档：

```bash
git rev-parse HEAD
git status --short
python --version
python -m pip freeze
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m morphoflow_ihc --help
python -m pytest -q
```

Windows PowerShell 可将 `python` 换成虚拟环境解释器，例如 `\.venv\Scripts\python.exe`。记录 GPU 型号、驱动、PyTorch/CUDA 版本、仓库提交号、配置文件哈希和数据 manifest 哈希。若本地安装环境未包含 `pytest`，先按 README 安装项目依赖再运行，不要把“模块缺失”记作通过。

## 2. 数据与配对审核

### 必须自动核对

- 每个 `pair_id` 恰有一张 H&E 和一张 IHC，marker 对应正确；路径存在且无重复复用造成的意外跨 split。
- H&E/IHC 的方向、色彩通道、像素尺寸和 MPP 元数据正确；统一到 `pixel_size_um` 后尺寸一致。
- 训练、验证、测试按患者 ID 分组。若只能按切片/来源前缀分组，明确标记为“患者隔离未证实”，不得写成 patient-independent split。
- 验证/测试图像不得被写入训练 crop cache；缓存记录和 manifest 哈希留档。

准备缓存：

```bash
python -m morphoflow_ihc prepare \
  --manifest datasets/MIST_HER2/manifest.csv \
  --config configs/mist_her2.json \
  --output data/preflight_cache
```

当前 `prepare` 会按来源前缀重新分组切分 MIST 清单，并且缓存仅写入训练对；它不能确认来源前缀等于患者 ID，也不能判断相邻切片是否真正配准。检查 `data/preflight_cache/index.json` 中的 `split_assignments`、`split_counts`、pair 数和 `settings`，同时与数据提供方的患者/切片元数据核对。

### 必须人工/图像质控

从训练、验证、测试各随机抽取一批配对图，查看整图和同坐标裁块。记录明显错位、组织缺失、折叠、空白区域和染色伪影比例；不要只挑好看的例子。用不依赖 DAB 表达的结构通道/特征估计局部位移，报告位移分布及无法可靠对应的比例。若结构无法在局部尺度对应，像素级 CFM 和配对 PatchGAN 目标就可能把错位当监督，应先解决配准或改用更粗粒度目标，再开始正式训练。

## 3. 自动测试与单步梯度检查

```bash
python -m pytest -q
```

测试应覆盖数据 split 泄漏、配对约束、CFM 条件输入、DAB 分解、PatchNCE 正负样本、奖励边界、策略动作对 RGB 三通道的有界作用、策略引导轨迹，以及生成器/critic/策略各自更新范围。任何失败都先修复并重新跑完整测试。

额外检查每一项生成器损失都能产生有限梯度：

```bash
python -m pytest -q tests/test_pipeline.py -k "patchnce or paired_reward_gradient or policy_guides_generator"
```

验收不是只看 loss 数值：CFM、PatchNCE、对抗项分别反传时，速度场参数都应出现有限非零梯度；PatchNCE 还应更新形态编码器/投影头；critic 更新时不应通过 detached fake 更新生成器；策略引导生成器步骤不应更新策略参数。

## 4. 奖励方向与策略可辨识性

执行顺序：先运行第 5 节的 `train` 命令生成 `runs/preflight/flow.pt`；再运行第 5 节的 `train-rl` 命令，并完成本节的奖励、策略输出和匹配对照检查；最后继续第 6 节推理检查。

### 受控奖励检查

在相同 H&E/参考 IHC 上构造只用于单元测试的可控输出：参考匹配、全白/无 DAB、额外 DAB、漏掉参考阳性，以及把阳性支持移到不同区域。检查奖励有界、排序符合设计，不能因为全局变浅就得到虚高分。现有自动测试至少覆盖“匹配参考优于空白输出”；正式开跑前还要检查额外染色和漏染两种方向。

### 真实验证集检查

只用训练完成后未参与梯度更新的验证配对，预先固定区域大小和 DAB 阈值。对每个 ROI 同时记录 reward、区域支持比例误差、局部 DAB 量误差，并检查 reward 是否与目标指标一致变化。若 reward 提高却系统性地让漏染/误染变差，不能放行 PPO；先修奖励或明确它优化的只是代理目标。测试集不得用于选择阈值、奖励权重或 checkpoint。

### 策略更新检查

策略从流模型 checkpoint 启动，先跑少量 PPO 组，确认日志中的 `reward`、`spread`、`accepted`、`kl` 全部有限；候选动作奖励差异不能全为零，KL 不得超过配置上限。对同一 H&E、同一 seed 分别用基础流 checkpoint 和策略 checkpoint 推理，确认策略确实改变了 RGB 输出；若短程内输出差异仍为零，应检查 PPO 更新是否被拒绝、速度场 RMS 是否接近零、动作是否饱和或被裁剪。策略变化本身不代表质量改善，仍需看验证指标。

再做一次小型匹配比较：从同一个 `flow.pt` 出发，普通流模型继续 8 次更新；策略模型进行 8 个样本组更新（每组包含两条候选轨迹）。两边使用同一训练/验证划分、seed、batch 和配置，分别在同一固定验证子集上计算区域支持比例误差、局部 DAB 量误差并保存配对图。它不是正式消融结论，只用于发现策略是否明显拉坏生成器或奖励与评价目标反向；如果短程结果噪声很大，记录为“未决”，不要据此选最佳配置。

```bash
python -m morphoflow_ihc train \
  --cache data/preflight_cache --config configs/mist_her2.json \
  --output runs/preflight/base_continue.pt --resume runs/preflight/flow.pt \
  --device cuda --steps 18 --batch 1 --workers 0 --save-every 1
```

这里 `--steps 18` 是包含已完成的前 10 步在内的目标总步数，因此该对照再训练 8 步；策略命令用 `--groups 8`。保留两条日志和固定验证子集清单。

## 5. CUDA 短程端到端预实验

使用训练 cache 和真实 CUDA 做小步数检查。短程权重单独放在 `runs/preflight/`，不得覆盖正式 checkpoint：

```bash
python -m morphoflow_ihc train \
  --cache data/preflight_cache --config configs/mist_her2.json \
  --output runs/preflight/flow.pt --device cuda \
  --steps 10 --batch 1 --workers 0 --save-every 1

python -m morphoflow_ihc train-rl \
  --cache data/preflight_cache --checkpoint runs/preflight/flow.pt \
  --output runs/preflight/policy.pt --device cuda \
  --groups 8 --batch 1 --workers 0 --save-every 1
```

若要试 batch 2，另建独立预实验目录，连续跑至少 100 次实际更新并记录峰值显存；不要仅凭一次前向或短暂 GPU 利用率决定正式 batch。正式消融要保持更新次数、样本暴露量、训练计划和 seed 一致；batch 不同会改变每次更新的梯度估计与优化轨迹，若确需混用，需作为配置差异报告并做敏感性检查。

记录：每步耗时、样本/秒、峰值显存、三项生成器损失、critic 损失、PPO reward/spread/KL/accepted 次数、梯度裁剪是否触发、checkpoint 是否能重载。10 步 smoke test 只验证管线和显存启动，不代表稳定收敛；放行前建议再跑至少数百次更新，检查滑动窗口损失、验证集指标及生成图是否持续出现模式崩塌、颜色漂移、过量/缺失 DAB 或 tile 接缝。

## 6. 推理与评价小检查

用验证集 H&E 测试基础流和策略流，固定 seed、MPP、marker、配置和 checkpoint。至少覆盖小于 tile、刚好 tile、非 tile 整数倍以及有重叠 tile 的图像；确认输出宽高等于输入原图，像素有限，边缘无未覆盖区域，重叠区无明显接缝。相同权重和 seed 应可复现；跨设备仅要求在预先定义的容差内一致。

```bash
python -m morphoflow_ihc infer --checkpoint runs/preflight/policy.pt \
  --image <validation_HE.png> --marker HER2 \
  --output runs/preflight/prediction.png \
  --maps runs/preflight/prediction.npz --device cuda --seed 1234

python -m morphoflow_ihc evaluate --prediction runs/preflight/prediction.npz \
  --reference <paired_validation_IHC.png> --marker HER2 \
  --config configs/mist_her2.json --output runs/preflight/metrics.json
```

不同切片的像素级比较只有在确认同层/配准可靠时才启用 `--same-section`。一般相邻切片应以区域代理指标为主，并在论文中说明错位限制；该代理不是细胞级 HER2/Ki-67 诊断。

## 7. 正式训练放行记录

正式训练前保存一份短报告，至少填写：

```text
Repository commit / dirty state:
Dataset manifest SHA-256 / pair count:
Patient-level split verified? If not, actual grouping unit:
MPP and canonical-size check:
Registration QC sample count / displacement summary / excluded fraction:
Environment, GPU, peak memory, throughput:
pytest result:
10-step flow smoke result and checkpoint reload:
PPO groups / finite reward spread / accepted updates / KL:
Base-vs-policy fixed-seed output difference:
Validation reward-vs-region-error direction:
Inference tiling/shape/reproducibility result:
Unresolved failures or caveats:
GO / NO-GO and reviewer:
```

**只有数据审核、完整测试、CUDA 短程训练、策略作用检查和验证奖励方向均通过，才进入正式训练。** 如果某项无法验证，明确记为未验证，不能以“代码能启动”替代。
