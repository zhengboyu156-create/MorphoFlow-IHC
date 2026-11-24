# MorphoFlow-IHC

**流匹配+强化学习**

Morphology-conditioned flow matching and single-step trajectory optimization for unpaired **H&E → virtual IHC**.

训练从真实 IHC 内部提取苏木素形态和 DAB。模型学习区域棕染概率，并将它作为单通道流场的软条件。推理时输入 H&E 和标志物，先生成 DAB，再在 H&E 原坐标上完成光密度合成。

## 数据与安装

MIST HER2 原始配对图位于 [datasets/MIST_HER2](datasets/MIST_HER2/README.md)，图片使用 Git LFS。数据授权和引用要求以原发布方为准。

使用 Python 3.10–3.12、PyTorch 2.8 或兼容构建。RTX 5070 Ti 可先安装官方 cu128 构建。CPU 支持用于功能检查。源项目的许可和固定版本见 [上游来源](docs/UPSTREAM.md)。

```bash
git clone --recurse-submodules https://github.com/zhengboyu156-create/MorphoFlow-IHC.git
cd MorphoFlow-IHC
python -m venv .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy Pillow scipy POT
python -m pip install --no-deps -e third_party/torchcfm ./third_party/torchdiffeq -e .
```

CPU 检查可把安装 PyTorch 的索引换成 `https://download.pytorch.org/whl/cpu`。安装命令选取上游模型、流匹配采样器和积分器实际需要的依赖。Flow‑GRPO 和 BranchGRPO 子模块用于源码参照。

准备一份 CSV。路径可以相对于 CSV。对同一患者使用相同的 `group`，包括不同标志物和染色。病人级编号不可用时填原切片组编号，评估独立性以实际分组范围为准。可选的 `mpp` 指各图的物理像素间距。

```csv
path,marker,group,domain,split,mpp
ihc/example-001.png,HER2,p001,IHC,train,0.4661
ihc/example-002.png,KI67,p002,IHC,train,0.4661
he/example-003.png,,p003,HE,test,0.4661
ihc/example-003.png,KI67,p003,IHC,test,0.4661
```

缓存命令检查分组，并仅打开标记为 `train,IHC` 的图片。真实配对文件可在留出评估时使用。示例配置里的标志物阈值是 **DAB 光密度代理阈值**，应按训练数据与染色规范定下来再锁定。

## 运行方法

以下命令使用同一配置和缓存。可以从已有检查点使用 `train --resume` 或以已有 RL 检查点继续 `train-rl`。时限在每次命令的步间检查，当前一步和检查点写出时间也应留在预算内。

```bash
python -m morphoflow_ihc prepare --manifest data/images.csv --config configs/mist.json --output data/cache
python -m morphoflow_ihc train --cache data/cache --config configs/mist.json --output runs/flow.pt --device cuda --hours 20
python -m morphoflow_ihc train-rl --cache data/cache --checkpoint runs/flow.pt --output runs/rl.pt --device cuda --hours 6
python -m morphoflow_ihc infer --checkpoint runs/rl.pt --image data/he/example-003.png --marker KI67 --output outputs/prediction.png --maps outputs/prediction.npz
python -m morphoflow_ihc evaluate --prediction outputs/prediction.npz --reference data/ihc/example-003.png --marker KI67 --config configs/mist.json
```

- 基础训练默认为 **25 轮、batch 4**。用 `--steps` 可指定更新步数。
- RL 默认为 **3,000 组**，每组有两个从同一前缀出发的候选。积分 8 步，第 5 步进行一次速度修正。训练采用时间上限和可恢复检查点。实际用时取决于 GPU、数据和存储吞吐。
- 推理接受 PNG/JPEG 等常见 RGB 栅格图像。输出 PNG 保持原始像素尺寸，NPZ 可保存 H、DAB 和区域预测图。用 `--mpp` 可提供不同于配置的像素间距。
- 评估默认报告同坐标窗口的 DAB 支持比例误差。确认是精确对齐的同切片数据时，`--same-section` 增加逐像素指标。配准由评估数据提供。

## 方法

| 部分 | 实现 |
| --- | --- |
| 区域预测 | 32 px 中心格读取局部核与组织特征，128 px 视野的外环提供背景。软标签 BCE 预测 DAB 面积比例。 |
| 条件流匹配 | 在噪声和真实 DAB 图之间构造直线路径，回归速度。条件为共享苏木素形态、区域软概率和标志物。 |
| 轨迹策略 | 固定基础流，学习有限幅度的区域速度残差。两个候选共享前缀，在一次随机动作后分别积分到终点。真实 IHC 评价漏染、额外染色和阳性轨迹的强度。 |
| 结构渲染 | 分离 H&E 的苏木素通道，按统一正增益校准。重叠窗口只融合预测 DAB，随后在整幅原始复染图上合成。 |

训练目标共三项，分别是区域预测、流匹配速度回归和轨迹策略优化。细节见 [方法与奖励](docs/METHOD.md)。从 IHC 复染学到的形态线索能否迁移到 H&E，需要独立的实图实验。复染底图的校准保留坐标和非量化强度顺序，最终视觉结构还受 DAB 遮盖影响。

## 功能验证

```bash
python -m pip install pytest
python -m pytest -q
```

测试的数据由程序绘制，包括合成 IHC 和 H&E，可验证流程与数值行为。项目面向研究，发布版本未包含训练数据、权重或临床性能结论。
