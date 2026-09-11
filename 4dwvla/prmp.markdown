# one
把要投RAL的论文 @4DW-VLA/4DW-VLA_1.markdown 按以下原则思路与要求改良一下:

## 发现的问题或痛点:
- 大部分VLA都是2D, 但机器人所在的实际是3D的, 甚至是4D(如果考虑时间维度上的变化的话)的世界.
- 有一些采用了3D信息的VLA,获取3D信息的成本很高,比如点云, 深度等信息都需要特定的设备或模型进行信息的抽取. 而且很多结合了3D信息的VLA又忽略了机器人在时间维度上的行为变化.
- 虽然预测的action是时间维度上的action, 但大部分VLA没有考虑到对2D或3D信息的时间维度的预测,很多了结合了世界模型做时间维度预测,但又忽略了3D信息.
- 而且机器人在时间维度上的动作, 状态, 整个3D时间的机器人本身的变化和周围环境的变化, 都是应该对齐的, 且都是互相联系的, 比如机器人在3D世界举起了手, 那么它的摄像头所看到的图像就应该包括了举起来的那只手.

## 我们的方法:
- 我们提出了4DW-VLA, 4D and World model aligned VLA. 以解决上面发现的问题和痛点. 
- 问题定义:
  * 对齐机器人实际活动的4D未来与机器人看到的2D未来:
    + 使用Unified Bidirectional Cross-Attention, 即两个transformer, 每个模块的每个时间步的token对应的embedding会在做attention的时候获取自己模块的该时间步之前的embedding的attention (self attention), 也会获取另一个模块该时间步前的embedding的attention (cross attention), 对于另一个模块的embedding也是同样的.
    + p_4D未来与2D未来 = (机器人在3D空间中活动的4D未来, 机器人摄像头看到的2D未来video | 当前摄像头图像, 机器人状态, 任务自然语言指令)
  * 预测机器人未来要做的动作
    + p_动作未来 = (机器人未来要做的多步动作 | 机器人在3D空间中活动的4D未来, 机器人摄像头看到的2D未来video, 当前摄像头图像, 机器人状态, 任务自然语言指令)

- 4D 未来专家(4D future expert)
  * 要突出为了避免获取4D信息的高成本, 我们选用了轻量级的FK方法计算link的3D关键点和姿态历史变化轨迹的4D信息获取方式, 比其它点云和深度图之类的4D信息获取快了几个数量级.
  * 7维(3D点+姿态)4D轨迹 = FK(机器人状态)
  * 预测4D未来的loss公式如下(要突出机器人状态q_t也是输入, 不仅为现在的4D未来预测提供信息, 也通过 corss attention为2D未来预测提供了信息):
$$
\mathcal{L}_{\mathrm{kpt}} = \beta \Bigl(
    \operatorname{MSE}\bigl(\hat{p}_t, p_t, q_t\bigr)
    + \gamma\, \operatorname{MSE}\bigl(\hat{p}_{t+1:t+H}, p_{t+1:t+H}, q_t\bigr)
\Bigr),
$$

- 机器人在3D空间的变化与摄像头能看到的变化进行对齐
  * 使用 world knowledge expert 基于机器人未来4D的变化来预测机器人在这种变化的情况下摄像头会看到什么样的video.
  * 对齐机器人实际活动的4D未来与机器人看到的2D未来:
    + 使用Unified Bidirectional Cross-Attention, 即两个transformer, 每个模块的每个时间步的token对应的embedding会在做attention的时候获取自己模块的该时间步之前的embedding的attention (self attention), 也会获取另一个模块该时间步前的embedding的attention (cross attention), 对于另一个模块的embedding也是同样的.
    + p_4D未来与2D未来 = (机器人在3D空间中活动的4D未来, 机器人摄像头看到的2D未来video | 当前摄像头图像, 机器人状态, 任务自然语言指令)
  * 通过用小 transformer 为世界模型提供机器人2D未来的隐式知识, 这种信息是已经与机器人未来4D的知识对齐了的, 使得世界模型能生成未来机器人摄像头会看到的video.
    + 这种设计也避免了庞大的世界模型作为VLA的一部分, 保证了模型的小型化与训练以及推理时的速度.
    + 关于机器人摄像头所能看到的未来2D变化的隐式知识 = world_knowledge_expert(机器人状态,当前机器人看到的图片,当前任务的自然语言指示, 4D_future_expert的隐式知识)
    + 未来机器人摄像头所能看到的未来2D的变化 = 世界模型(关于机器人摄像头所能看到的未来2D变化的隐式知识) , 而这个庞大的世界模型在推理时是可以去掉的, 因为对未来的预测和知识都已经在`未来2D变化的隐式知识`中了.
    + 庞大的世界模型在训练时是冻结的, 在推理时是去掉的, 保证了训练和推理的速度, 以及整个VLA模型的小型化.

- 多模态多任务共同训练
  * 输入模态有: 图像模态, 文本模态, 机器人状态, 4D模态
  * 共同训练的任务有: 
    + 基于图像模态和文本模态, 同时也基于4D的历史预测预测当前机器人的3D状态
    + 基于图像模态和文本模态, 同时也基于机器人状态与4D历史和计算出来的机器人当前的3D状态预测未来的4D. 在做这个任务的时候会通过`Unified Bidirectional Cross-Attention`用到`未来2D变化的隐式知识`.
    + 基于图像模态和文本模态, 同时也基于机器人状态与4D历史和计算出来的机器人当前的3D状态预测未来的2D video. 在做这个任务的时候会通过`Unified Bidirectional Cross-Attention`用到`4D_future_expert的隐式知识`.
    + Action Expert 去预测未来的action chunk. 需要生成一个 loss 公式.
  * 需要生成一个多任务的 loss 公式, 而且不同子loss间可通过权重协调.

## 改论文时其它注意的点
- 先把实验部分去掉, 生成剩余部分的内容和公式.
- 不要提及InternVLA, GeoPredict, ELAN4D 等其它利用了 FK 去计算 3D 信息的VLA.
- 在`Related Work`中的`3D and 4D Representations for Manipulation`章节中, 只从 @4dwvla\ls_4dwrk.md 挑一些大公司出的论文进行讨论, 重点是指出它们用的4D信息太重,占计算量,导致模型尺寸也大.

---

基于上述原则思路与要求改良了 4DW-VLA_1.markdown 之后, 写到 @4dwvla/4DW-VLA_2.markdown 中.

# Two

把要投RAL的论文 @4DW-VLA/4DW-VLA_1.markdown,  @4dwvla/4DW-VLA_2.markdown 是它的改良后的版本, 但后者还缺实验部分, 按以下原则思路与要求改良一下:

## 实验部分的修改
- 不要提4DW-VLA是从InternVLA改良过来的. GeoPredict 一点都不能提.
- 只说 InternVLA A1.5 是对比基线的一部分, 代表了考虑了时间和未来机器人能看到的2D video的预测,但没考虑3D/4D信息的那一类VLA
- LingBotVLA v2 也是对比基线的一部分, 代表了考虑了 3D(深度) 和 时间变化(video生成) 但没考虑3D,4D,时间上的2D变化之间的对齐的那一类VLA. 

## 其它修改
- 把warmup这一步去掉, 但要训练时世界模型部分是冻结的, VLM部分的学习率只有其它部分的学习率的1/5.
- 把训练的76 epoch 改成 10000 steps

---

基于上述原则思路与要求改良了 4DW-VLA_1.markdown 的实验部分后加到 4DW-VLA_2.markdown 中. 同时也基于上述原则思路与要求再检查和改良一下 `4DW-VLA_2.markdown` .




warmup 用 400 steps, 或者直接不写 warmup.

先把实验去掉. 剩下的部分不要提 InternVLA 和 GeoPredict 以及其它利用了 FK 3D 的VLA.
再写实验部分.
    InternVLA 属于考虑了时间,但没考虑3D/4D信息.
    LingBotVLA 属于考虑了 3D(深度) 和 时间变化(video生成) 但没考虑3D,4D,时间上的2D变化之间的对齐.
 