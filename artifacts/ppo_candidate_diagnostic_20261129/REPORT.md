# 多来源 PPO 候选动作诊断与 1,000 组 pilot

日期：2026-11-29；分支：`run/preformal-efficacy-20261128`；基线代码提交：`b6a500e715f87d5dd707fae570c4f775de7a5361`。

## 结论

把开发集覆盖从 2 个来源前缀扩到 61 个后，PPO 在 `velocity_limit=1.0` 的探索性训练中，候选组信号持续有效；在固定的 128 对 pilot-dev 上，策略开启结果优于同起点零动作续训及同 checkpoint 关策略的结果。不过，这个开发集已用于比较速度修正尺度并选出 1.0，存在开发集适配/选择偏差。因此结果只能说明 PPO 值得进入独立冻结后的验证，不能作为确认性效能或论文结论。

本轮未碰官方测试集、未启动完整 40+10 训练。患者身份未核实，61 个 `source_prefix` 不是已确认患者 ID。所有生成权重留在本机；Git 中只归档代码、配置、运行日志、清单哈希及数值结果。

## 数据与匹配设计

- 基础来源：MIST 官方训练 split 4,642 对；pilot 划分只在这批训练数据中进行，不使用官方测试 split。
- 固定划分：pilot-train 4,514 对，pilot-dev 128 对。pilot-dev 覆盖全部 61 个可用 `source_prefix`（每前缀 1–3 对）；它不是患者独立性证明。
- 划分 seed：`20261129`。split JSON SHA-256：`478832c368d1cc8658a5a37e4f60877e49fa18d72ab424066b8039a03ff57591`。
- train cache index SHA-256：`65cb98d240d32de7c8bf33f5e0a4d8d205dd2ea1fd9fd2b4997d82684bf9c8ba`；dev cache index SHA-256：`d9e63951a6ea2f595aeacc394567d237210cc76b474d317a49b2d37e9908cc73`。
- Flow、PPO、零动作各使用 1,000 次生成器更新；相同 pilot-train、seed `20261129`、batch 1、BF16 AMP、10 epoch 学习率衰减设置。PPO 每状态 4 候选；低信号门槛保持 `min_reward_spread=1e-5`，未调低。
- 评估使用同一对图像的确定性 512 crop 和同一高斯噪声，FP32 推理；128 对、61 个来源前缀。表中的区间为来源前缀等权 bootstrap 95% 描述区间，重采样单位不是患者。

## PPO 训练信号与候选动作

1,000 PPO 组完成，日志采样点 `signal_fraction=1.0` 且 `accepted=2`；最后记录的 reward spread 为 `0.005504`，高于原门槛。候选动作诊断同时记录了动作范数、速度修正、奖励分量、策略开关图像差和输出范围裁剪情况，详见 `candidates_flow1000_integration/summary.json` 与 `candidate_actions.csv`。

初始 `velocity_limit=0.1` 时，候选策略输出变化很弱。无权重更新的尺度扫描显示，提高到 1.0 会增强速度修正、输出差异和组内 reward 排序。于是新跑了一条从同一 Flow1000 初始化的 PPO 训练臂，唯一主动变化为策略速度修正上限。尺度是看过 pilot-dev 后选择的，所以本次 dev 比较偏探索性。

## 128 对 pilot-dev 指标

支持比例 MAE 越低越好；support F1 使用当前 1% 阳性窗口阈值；DAB 总量相对误差越低越好。

| 模型 | 64 px support MAE | 128 px support MAE | support F1 (64/128) | DAB 总量相对误差 | 参考组织内平均 DAB |
|---|---:|---:|---:|---:|---:|
| Flow1000 | 0.32147 | 0.29826 | 0.75015 / 0.78915 | 2.60226 | 0.81206 |
| PPO velocity limit 1.0，策略开 | **0.28236** | **0.26069** | 0.75015 / 0.78915 | **1.27362** | **0.48250** |
| 同一 PPO checkpoint，策略关 | 0.29251 | 0.27011 | 0.75015 / 0.78915 | 1.47612 | 0.53440 |
| 零动作续训1000 | 0.29553 | 0.27271 | 0.75015 / 0.78915 | 1.61097 | 0.56904 |

PPO-on 减零动作的 128 px support MAE 平均配对差为 `-0.01202`，来源前缀等权均值 `-0.01260`，描述性 95% 区间 `[-0.01723, -0.00780]`（61 个来源前缀中 44 个为负、17 个为正）。DAB 总量相对误差差为 `-0.33735`，来源前缀等权 95% 区间 `[-0.40684, -0.27543]`。

策略开相对同 checkpoint 关策略，128 px support MAE 差为 `-0.00942`；DAB 总量相对误差差为 `-0.20250`。成对 RGB01 MAE 为 `0.00875`，表示策略改变的输出量已可测，但仍属于幅度不大的修正。

所有模型的 support F1 完全相同，说明当前 1% 窗口阈值对本轮差异不敏感，不能把它写成 PPO 改善证据。PPO-on 的 DAB 总量相对误差仍为 1.27，即估计的总量误差仍大，当前模型不应被称为已校准或可正式放行。

## 限制与下一步

1. 速度尺度由同一 pilot-dev 筛选，区间不能消除尺度选择造成的偏差。需要锁定参数后，在未参与选择的验证数据上复验；若没有额外独立开发划分，结论继续保持探索性。
2. `source_prefix` 是文件来源分组，不是患者 ID；切片/患者独立性仍未知。
3. support F1 的阈值需要先做不看模型结果的灵敏度/校准分析，随后固定，不能事后挑有利阈值。
4. 本轮只直接验证 PPO。区域 CFM 的位移压力测试、PatchNCE 结构代理、PatchGAN 染色外观护栏仍未完成，不能从 PPO 结果推断它们有效。
5. Flow 和两条续训臂均处于短程 pilot；DAB 误差仍高。下一步应先冻结动作尺度和主指标，再用未参与筛选的开发集做一次匹配复验；不直接宣称完整训练获批。

## 可复核文件

- 训练日志：`logs/pilot_flow1000.log`、`logs/ppo1000.log`、`logs/zero_action1000.log`、`logs/ppo1000_velocity1.log`
- 候选尺度扫描：`velocity_scale_sweep/summary.json`、`velocity_scale_sweep.csv`
- 本次最终评估：`evaluation128_velocity1/summary.json`、`evaluation128_velocity1/per_pair_metrics.csv`
- 前一版 0.1 策略尺度对照：`evaluation128_complete/summary.json`、`evaluation128_complete/per_pair_metrics.csv`
- 代码、数据和权重哈希清单：`provenance.json`
