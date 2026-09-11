# 4D VLA / Policy 相关工作文献（深度、点云等非 FK 几何）

> 范围：使用 **深度图、点云、RGB-D、3D Gaussian、场景轨迹场** 等感知/重建得到的 3D/4D 信息来增强 VLA 或 manipulation policy 的论文。  
> **排除**：主要依赖 **FK 正运动学** 计算机器人关节 3D keypoint 轨迹作为核心几何信号的 work（如 ELAN4D、GeoPredict 等）。  
> 排序：**由新到旧**。

---

## 1. 4D-WAM: Infusing Spatiotemporal Awareness into World Action Models through Trajectory Fields

- **网址**：https://arxiv.org/abs/2608.08023
- **简介**：面向 World-Action Model（WAM）的模型无关训练策略，利用 4D 基础模型 Trace Anything 从 RGB 视频估计的 **像素级 3D 轨迹场**（trajectory fields），在训练期对齐 WAM 中间视觉特征，推理时丢弃 teacher，无额外开销。
- **创新点**：
  - 将每个像素跨帧关联到参数化 3D 轨迹，构成显式 4D 表示（3D 位置随时间变化）。
  - **Motion Alignment**：对齐相邻帧特征变化与 3D 状态转移，捕获局部运动动力学。
  - **Destination Alignment**：从源帧推断目标区域，增强长程操作目标感知。
  - 不修改 WAM 架构，仅通过表征对齐注入时空感知。

---

## 2. Lift3D-VLA: Lifting VLA Models to 3D Geometry and Dynamics-Aware Manipulation

- **网址**：https://arxiv.org/abs/2607.06564 | https://lift3dvla.github.io/
- **简介**：统一 VLA 框架，将 **显式点云** 编码进预训练 2D VLA 视觉塔，并通过自监督学习同时建模当前几何与未来几何演化，实现几何感知 + 时序一致的动作生成。
- **创新点**：
  - **Lift3D 投影**：将点云投影到多个虚拟平面，与 2D 位置编码对齐，最小化 3D→2D 编码信息损失。
  - **GC-MAE（Geometry-Centric Masked Autoencoding）**：重建当前点云 + 预测未来点云几何，使 2D encoder 内化 3D 结构与物理动力学。
  - **Layer-wise Temporal Action Modeling**：利用 LLM 多层序列表示协同预测 action chunk，提升时序连贯性。
  - 点云来自 VGGT 等从 RGB 轨迹生成的伪标签，无需 FK。

---

## 3. RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation

- **网址**：https://arxiv.org/abs/2607.06559 | https://alibaba-damo-academy.github.io/RynnWorld-4D.github.io/
- **简介**：4D 具身世界模型，从单帧 **RGB-D + 语言指令** 在统一扩散过程中联合生成未来 RGB、**深度图** 与 **光流**（RGB-DF），并据此训练 **RynnWorld-4D-Policy** 逆动力学头做闭环双臂操作。
- **创新点**：
  - **RGB-DF 投影 4D 表示**：深度将像素提升到 3D，结合光流可反投影为 3D scene flow，显式编码几何与运动。
  - **Tri-branch Diffusion**：RGB / depth / flow 三分支 + 跨模态注意力 + frame-wise 3D RoPE，保证时空一致。
  - **Action-from-Latent**：Policy 直接消费世界模型内部 4D latent，单次前向输出动作，跳过多步去噪，支持 9Hz+ 闭环控制。
  - 大规模 Rynn4DDataset（2.54 亿帧，含伪标注 depth/flow）。

---

## 4. MotionVLA: Injecting Geometric Motion into Vision-Language-Action Model

- **网址**：https://arxiv.org/abs/2606.08288
- **简介**：提出 **motion-history interface**：用冻结 Trace Anything 将短历史 RGB 窗口转为紧凑、时间连续的 **trajectory-field tokens**，当前视觉 token 从中检索任务相关运动证据，再耦合回 VLA 流，改善长程操作。
- **创新点**：
  - 核心问题：VLA 应记住“帧之间的运动”而非“更多独立帧”。
  - 场景 **trajectory field**（由视频 3D 重建/跟踪得到）作为可查询运动记忆，非 FK 关节 keypoint。
  - Trajectory-grounded supervision 抑制几何漂移与动作不稳定。
  - 与 DepthVLA、GeoVLA、PointVLA 等静态几何方法形成互补（强调时序一致运动证据）。

---

## 5. PointAction: 3D Points as Universal Action Representations for Robot Control

- **网址**：https://arxiv.org/abs/2606.03943 | https://oriontmt.github.io/pointaction/
- **简介**：Video-Action 框架，微调基础视频扩散模型联合预测未来 RGB 与 **动态 3D pointmap**（由深度/重建得到），再以 embodiment-specific 扩散解码器将点轨迹映射为机器人动作。
- **创新点**：
  - **Universal video-to-point + specific point-to-action** 两阶段分解，跨本体泛化。
  - 3D point dynamics 作为视频预测与控制之间的 **embodiment-agnostic 中间接口**。
  - RGB 与 XYZ pointmap VAE latent 拼接，DiT 内 patch 级几何-语义对齐。
  - PointNet 编码 robot-centric 点云，条件 DiT 解码 action chunk。

---

## 6. ConsisVLA-4D: Advancing Spatiotemporal Consistency in Efficient 3D-Perception and 4D-Reasoning for Robotic Manipulation (CVPR 2026)

- **网址**：https://arxiv.org/abs/2605.05126 | https://github.com/JiuTian-VL/ConsisVLA-4D
- **简介**：OpenVLA 风格的高效 4D VLA，从 **2D 观测** 高效生成 3D 表征并增强 4D 时空推理一致性，无需额外深度传感器推理开销。
- **创新点**：
  - **CV-Aligner**：跨视角过滤指令相关区域，对齐物体语义身份。
  - **CO-Fuser**：跨物体空间几何一致性，消除多视角空间关系歧义。
  - **CS-Thinker**：结合语义 token 与 **几何 depth token**，在动作执行过程中保持跨场景时空一致。
  - 在 LIBERO 与真机平台上相对 OpenVLA 显著加速（约 4.1×）且成功率更高。

---

## 7. PointNet4D: A Lightweight 4D Point Cloud Video Backbone for Online and Offline Perception in Robotic Applications (WACV 2026)

- **网址**：https://doi.org/10.1109/wacv61042.2026.00313 | https://github.com/yunzeliu/MAP
- **简介**：轻量 **4D 点云视频** backbone（Hybrid Mamba-Transformer 时序融合 + 4DMAP 预训练），并据此构建 **4D Diffusion Policy (DP4)** 与 **4D Imitation Learning (4DIL)** 机器人应用。
- **创新点**：
  - 面向在线/离线流式点云视频，参数与算力开销极低（RoboTwin 上仅 +0.3% 参数）。
  - **4DMAP**：帧级 masked auto-regressive 预训练，捕获跨帧运动线索。
  - DP4 在 RoboTwin 14 项任务、4DIL 在 HandoverSim 上显著优于 prior diffusion/IL 方法。
  - 直接以 **传感器/重建点云序列** 为策略输入，非 FK 轨迹。

---

## 8. Pri4R: Learning World Dynamics for Vision-Language-Action Models with Privileged 4D Representation

- **网址**：https://arxiv.org/abs/2603.01549 | https://jiiiisoo.github.io/Pri4R/
- **简介**：训练期利用 **privileged 场景 3D point tracks**（从深度/跟踪得到的高保真 4D 几何）作为辅助监督，引导 VLM latent 编码动作-世界因果动态，推理零额外开销。
- **创新点**：
  - 专用 point-tracking head 联合预测动作与未来 **场景点轨迹**（非机器人 FK keypoint）。
  - 实验证明 3D point tracks 优于纯 depth 监督（depth 缺乏跨帧点身份，静态场景冗余大）。
  - 在 RoboCasa（+40%）与 LIBERO-Long（+10%）显著提升。
  - 保持标准 VLA 接口，privileged 信号仅训练期使用。

---

## 9. ST-VLA: Enabling 4D-Aware Spatiotemporal Understanding for General Robot Manipulation

- **网址**：https://arxiv.org/abs/2603.13788 | https://oucx117.github.io/ST-VLA/
- **简介**：层次化 VLA，用统一 **3D-4D 表征** 桥接语义推理与连续控制；配套 ST-Human 数据集（30 万 episode，含 **depth、物体几何、时序动作演化** 标注）训练 ST-VLM。
- **创新点**：
  - 将 2D 引导提升为 **3D 轨迹**，并生成平滑空间 mask 编码 4D 上下文。
  - ST-VLM 输出 spatially grounded、temporally coherent 的 3D 表征指导 policy。
  - 支持在线重规划与长程执行；RLBench 零样本成功率 +44.6%，真机 +30.3%。
  - 几何来自 **深度感知与 3D 观测**，非 FK 关节轨迹。

---

## 10. DepthVLA: Enhancing Vision-Language-Action Models with Depth-Aware Spatial Reasoning

- **网址**：https://arxiv.org/abs/2510.13375
- **简介**：Mixture-of-Transformers VLA，引入预训练 **depth prediction expert**，与 VLM、action expert 全共享注意力，端到端增强空间推理，无需显式点云输入或专用深度传感器。
- **创新点**：
  - Depth expert 提供细粒度几何 cue，VLM 提供语义，action expert 条件于双模态表征。
  - 各组件可独立预训练再融合，提升训练效率。
  - 真机任务 progress 78.5% vs 65.0%，LIBERO 94.9%，Simpler 74.8%。
  - 深度由网络预测而非 FK 计算。

---

## 11. GeoVLA: Empowering 3D Representations in Vision-Language-Action Models

- **网址**：https://arxiv.org/abs/2508.09071 | https://linsun449.github.io/GeoVLA/
- **简介**：并行处理 RGB-L 与 **depth→point cloud**，经 Point Embedding Network (PEN) 生成 3D 几何 embedding，再由 3D-enhanced Action Expert (3DAE) 融合多模态输出动作。
- **创新点**：
  - 深度图转点云 + 定制点编码器，与 VLM 2D 语义 embedding 拼接。
  - 在 LIBERO、ManiSkill2 达 SOTA，真机展现高度/尺度/视角鲁棒性。
  - 端到端利用 **感知深度** 而非机器人运动学 keypoint。

---

## 12. 4D Diffusion Policy (DP4): Spatial-Temporal Aware Visuomotor Diffusion Policy Learning (ICCV 2025)

- **网址**：https://arxiv.org/abs/2507.06710 | https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_Spatial-Temporal_Aware_Visuomotor_Diffusion_Policy_Learning_ICCV_2025_paper.pdf
- **简介**：4D 扩散策略，从单视角 **RGB-D** 构建当前 3D 场景，用 **dynamic Gaussian world model** 预测未来 3D 场景，显式建模空间-时间依赖以指导轨迹生成。
- **创新点**：
  - Dynamic Gaussian Splatting 世界模型：deformable MLP 传播 Gaussian 参数，渲染未来 RGB-D。
  - 将 4D 时空感知注入 diffusion policy，超越纯 behavior cloning。
  - 17 个仿真任务（173 变体）+ 3 个真机任务验证；Adroit +16.4%、DexArt +14%、RLBench +6.45%。
  - 几何完全来自 **RGB-D 观测与 Gaussian 重建**。

---

## 13. GAF: Gaussian Action Field as a 4D Representation for Dynamic World Modeling in Robotic Manipulation

- **网址**：https://arxiv.org/abs/2506.14135
- **简介**：基于 3D Gaussian Splatting 的 **V-4D-A** 实现：从 RGB 视频（无需 GT 3D）学习可微渲染的 **Gaussian Action Field**，输出当前/未来 Gaussian 场景并通过点云配准得到初始动作。
- **创新点**：
  - 每个 Gaussian 带 learnable motion attribute，编码时序位移，统一表示 evolving scene geometry。
  - **Action-vision-aligned denoising** 模块用 GAF 视觉引导 refine 初始动作。
  - 仅需 RGB 视频监督（通过可微渲染），无需 FK 或外部 3D 标注。
  - 连接动态视觉感知与动作生成的完整 V-4D-A pipeline。

---

## 14. 4D-VLA: Spatiotemporal Vision-Language-Action Pretraining with Cross-Scene Calibration (NeurIPS 2025)

- **网址**：https://arxiv.org/abs/2506.22242 | https://papers.neurips.cc/paper_files/paper/2025/file/30b9c38b9ebeee281cd2bc41d39bf0e7-Paper-Conference.pdf
- **简介**：将 **4D 信息（depth + 时间）** 注入 VLA 预训练，用 sequential **RGB-D** 输入与 3D coordinate embedding 对齐机器人与场景坐标系，缓解 cross-scene pretraining 中的 coordinate/state chaos。
- **创新点**：
  - 3D-aware module 生成 spatial vision tokens，统一多场景坐标系。
  - **Memory Bank Sampling**：从历史帧中自适应选取信息量最大的帧，提升效率。
  - 提出 MV-Bench 多视角仿真 benchmark 评估空间泛化。
  - 相对 OpenVLA 在仿真与真机均显著提升成功率。

---

## 15. FP3: A 3D Foundation Policy for Robotic Manipulation (ICRA 2026 Best Paper)

- **网址**：https://arxiv.org/abs/2503.08950
- **简介**：**3D 基础策略**，以 **点云** 为核心 3D 表征，在大规模多任务数据上预训练，可高效迁移到新任务/新机器人。
- **创新点**：
  - 将点云作为 manipulation 的 universal 3D substrate，兼顾泛化与样本效率。
  - 3D foundation pretraining + 下游 fine-tuning 范式。
  - 与 DP3 等同属 **depth/点云驱动** 的 3D policy 线，但强调 foundation-scale 预训练。
  - 点云通常由 RGB-D 或深度传感器反投影得到。

---

## 16. PointVLA: Injecting the 3D World into Vision-Language-Action Models

- **网址**：https://arxiv.org/abs/2503.07511
- **简介**：在不重训整个 VLA 的前提下，冻结 action expert，通过 **skip-block 分析** 定位可注入层，用轻量 modular block 将 **LiDAR/深度点云** 特征注入预训练 VLA。
- **创新点**：
  - 最小化对 2D 预训练表征的干扰，仅向“较不重要”的 block 加性注入 3D 特征。
  - 适用于 3D 机器人数据远小于 2D 预训练数据的场景。
  - 保留 VLM backbone  intact，低成本获得点云条件策略。
  - 3D 输入来自 **传感器点云**，非 FK。

---

## 17. 3D Dynamics-Aware Manipulation: Endowing Manipulation Policies with 3D Foresight

- **网址**：https://arxiv.org/abs/2502.10028 | https://github.com/Stardust-hyx/3D-Foresight
- **简介**：将 **3D 世界建模** 与 policy learning 结合，通过三项自监督任务——当前 **depth 估计**、未来 **RGB-D 预测**、**3D flow 预测**——赋予策略 3D foresight，推理速度不受影响。
- **创新点**：
  - 针对 depth-wise 运动显著的任务，2D 动态建模不足的问题。
  - 3D 相关目标（depth、RGB-D、3D flow）互补，显著优于 2D foresight。
  - 仿真与真机验证；建议未来扩展至 point cloud、Tri-Plane、3DGS 等更高级 3D 表征。
  - 深度/flow 均由网络从 RGB 预测。

---

## 18. SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model

- **网址**：https://arxiv.org/abs/2501.15830 | https://spatialvla.github.io/
- **简介**：在 110 万真实机器人 episode 上预训练的 spatial-enhanced VLA，用 **ZoeDepth 估计深度** 并 **Ego3D Position Encoding** 将像素反投影为相机坐标系 3D 点，与 2D 语义特征融合；动作用 Adaptive Action Grids 离散化。
- **创新点**：
  - Egocentric 3D position encoding 消除对外参/多相机标定的强依赖。
  - Adaptive Action Grids 统计动作分布、支持跨机器人迁移与高效 fine-tune。
  - LIBERO-Spatial 88.2%，OOD WidowX 任务显著优于无深度 VLA。
  - 明确论证：集成 **depth/点云** 对空间布局变化鲁棒性至关重要。

---

## 19. 3D Diffusion Policy (DP3)

- **网址**：https://arxiv.org/abs/2406.04416（RSS 2024）
- **简介**：经典 **3D diffusion policy**，以简单 **点云**（通常由 RGB-D 下采样得到）为观测，证明相对 2D image policy 在泛化与样本效率上的优势。
- **创新点**：
  - Minimalist 3D 表征：无需复杂 3D backbone 或额外传感器（仿真 depth 即可）。
  - 点云 + diffusion 生成 action trajectory，成为后续 FP3、PointVLA 等工作的基线。
  - 开启“3D substrate for visuomotor policy”主流路线。
  - 几何完全来自 **深度/点云感知**，与 FK 无关。

---

## 20. Act3D: 3D Feature Field Transformers for Multi-Task Robotic Manipulation (CoRL 2023)

- **网址**：https://arxiv.org/abs/2309.12235
- **简介**：用 **3D feature fields**（由 depth 反投影 + 多视角聚合的 3D 体素/特征场）作为 Transformer policy 的统一观测，支持多任务 manipulation。
- **创新点**：
  - 将 2D 预训练特征提升到 3D 空间，在 3D query 位置聚合。
  - 单一架构处理 pick-place、articulated object 等多任务。
  - 依赖 **depth-based 3D lifting**，非机器人 FK 轨迹。
  - 影响后续 3D feature field / voxel policy 系列工作。

---

## 21. Perceiver-Actor (PerAct): A Multi-Task Transformer for Robotic Manipulation (CoRL 2022)

- **网址**：https://arxiv.org/abs/2209.05451
- **简介**：早期代表性 **3D manipulation** 方法：将多视角 RGB-D 体素化为 **voxel grid point cloud**，经 Perceiver IO 编码后预测 discretized 6-DoF pose + gripper action。
- **创新点**：
  - 显式 **voxelized 3D 场景表示** 替代纯 2D CNN policy。
  - Language-conditioned multi-task transformer actor。
  - 3D 体素由 **深度图反投影** 构建，是 depth-driven 3D policy 的奠基工作之一。
  - 为后续 3D/4D VLA 的“几何显式化”思路提供早期范式。

---

## 附录：与本清单的区分说明

| 类型 | 代表工作 | 几何来源 | 是否列入 |
|------|----------|----------|----------|
| FK 关节 keypoint 4D 监督 | ELAN4D, GeoPredict（kinematic 分支） | 正运动学 | **否** |
| 场景点轨迹 / 像素轨迹场 | Pri4R, 4D-WAM, MotionVLA | 深度跟踪 / Trace Anything | **是** |
| 点云 / RGB-D 直接输入 | DP3, FP3, Lift3D-VLA, PointVLA | 传感器或 depth 重建 | **是** |
| 3D Gaussian / 动态高斯 | GAF, DP4 | RGB(-D) 可微重建 | **是** |
| 预测深度（无额外传感器） | DepthVLA, SpatialVLA, ConsisVLA-4D | 单目 depth 估计 | **是** |

---

*最后更新：2026-09-10*
