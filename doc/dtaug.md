
20.50KB
实施方案：ImageTransformsConfig 顶层概率门控 p（修订版）​
目标：为图像数据增强流水线增加一个顶层概率参数 p，控制"是否应用数据增强"的概率。p=0.8 表示 80% 的样本经过数据增强，20% 完全不增强。 一致性保证：同一 sample 的所有相机、所有帧共享同一个增强/跳过决策。 基于: 当前 dtaug2 实验（全部 6 种增强 + affine@0.2）的代码和配置 日期: 2026-09-23​
​
​
问题分析​
1.1 现状​
当前 RandomSubsetApply.forward() 在 transforms.py:74-87 中无条件采样并应用 n_subset 个变换，没有"跳过增强"的路径。只要 enable=True 且有可用变换，每张图片 100% 被增强。​
1.2 帧内一致性的现状​
已有保证：torchvision v2 的 RandomAffine 对 [T, C, H, W] batch tensor 生成一组参数并应用到所有 T 帧。实验验证如下：​
​
uniform = torch.ones(4, 3, 64, 64) * 0.5​
tf = v2.RandomAffine(degrees=(-5, 5), translate=(0.05, 0.05))​
out = tf(uniform)​
torch.allclose(out[0], out[1])  # True — 所有帧同一变换​
torch.allclose(out[0], out[3])  # True​
​
因此，同一 self.image_transforms(item[cam]) 调用中的所有帧（当前帧 + delta 帧）已经共享相同的 affine 参数。​
未保证：多相机场景下，每个 camera 独立调用 forward()，增强决策独立。如果 p 门控放在 forward() 内，同一 sample 的不同相机可能一个增强一个不增强。​
1.3 目标​
引入参数 p ∈ [0.0, 1.0]，语义为 per-sample 开关：​
在每个 sample 的 __getitem__ 开始时，掷一次骰子决定"增强"或"跳过"​
该决策对该 sample 的所有相机、所有帧生效​
不同 sample 之间独立决策​
一致性矩阵：​
​
范围​
一致性​
由谁保证​
同一 camera 的多帧​
✅ 相同 affine 参数​
torchvision v2 batch 行为​
同一 sample 的多 camera​
✅ 相同增强/跳过决策​
本方案 begin_sample() 机制​
不同 sample​
❌ 独立随机​
预期行为​
​
1.4 为什么不做 per-batch​
PyTorch DataLoader 中 __getitem__ 由多个 worker 进程独立调用，每个 worker 不知道自己属于哪个 batch。真正的 per-batch 需要修改训练循环（在 collate 之后、forward 之前做增强），改动量大且侵入性强。​
per-sample 已经满足核心需求：同一 sample 内所有视觉数据的增强决策一致。不同 sample 间的独立性反而有利于训练——如果整个 batch 统一跳过，会导致该 step 的梯度完全来自"无增强"分布，造成梯度信号不均匀。​
​
​
​
文件​
改动​
性质​
src/lerobot/datasets/transforms.py​
配置加字段 + begin_sample() + forward() 门控​
核心​
src/lerobot/datasets/lerobot_dataset.py​
__getitem__ 中调用 begin_sample()​
适配​
src/lerobot/datasets/streaming_dataset.py​
同上​
适配​
​
​
保证​
描述​
机制​
✅ 同相机所有帧：相同 affine 参数​
torchvision v2 对 [T,C,H,W] 生成一组参数​
已有​
✅ 所有相机：相同增强/跳过决策​
begin_sample() 缓存决策​
本方案​
⚠️ 不同相机：affine 参数不同​
独立的 forward() 调用，独立的随机参数​
已有行为，未改变​
✅ 不同 sample：独立随机​
独立的 begin_sample() 调用​
预期行为​
​
​
场景​
p 值​
begin_sample() 被调用?​
行为​
与当前对比​
默认（不传 p）​
1.0​
是，但 p>=1.0 短路​
100% 增强​
完全一致​
现有 dtaug2 训练​
1.0​
是​
100% 增强​
完全一致​
增强关闭 (enable=false)​
任意​
image_transforms 为 None​
不增强​
完全一致​
旧代码不调 begin_sample()​
任意​
否​
_augment_this_sample=True（默认）​
完全一致​
​
​
参数组合​
语义​
每图增强概率​
每图 affine 概率​
p=0.8, max_num_transforms=3, 6 种变换, affine@0.2​
80% sample 增强，其中 ~11.1% 含 affine​
80%​
≈ 8.9%​
p=1.0（默认）​
与当前 dtaug2 完全一致​
100%​
≈ 11.1%​
​
​
参数​
dtaug2​
dtaug2p​
p (增强概率)​
1.0 (隐式)​
0.8​
6 种增强​
✅ 全开​
✅ 全开​
affine weight​
0.2​
0.2​
max_num_transforms​
3​
3​
per-sample 一致性​
N/A​
✅ begin_sample()​
起点 checkpoint​
SFT@10420​
待定 (见 5.2)​
训练步数​
20000​
20000​
其余参数​
—​
完全一致​
​
20% 的 sample 不做任何增强，为模型提供"真实分布锚点"​
同一 sample 内所有相机、所有帧的增强决策完全一致​
验证指标：LIBERO / RoboTwin eval 成功率​
配置定义: src/lerobot/configs/default.py:37​
torchvision v2 RandomAffine batch 行为验证: 本文档 §1.2 实验