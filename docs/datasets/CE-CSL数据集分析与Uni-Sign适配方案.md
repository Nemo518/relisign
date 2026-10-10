# CE-CSL 数据集分析与 Uni-Sign 适配方案

## 1. 目标

这份文档用于解决两件事：

1. 梳理你当前本地 `CE-CSL` 数据集的真实目录结构和标注格式
2. 设计 Uni-Sign 接入 `CE-CSL` 的代码改造方案

当前分析的数据集路径为：

```text
E:\project\xiaozhi\xiaozhi-esp32-2.2.2\xiaozhi-esp32-2.2.2\ai-models\data\CSL-500\CE-CSL\CE-CSL
E:\project\xiaozhi\Uni-Sign\data\CE_CSL\CE-CSL
```

## 2. CE-CSL 目录结构分析

### 2.1 顶层目录

当前 `CE-CSL` 顶层包含以下 5 个子目录：

1. `extracted_frames`
2. `frames`
3. `label`
4. `label_cleaned`
5. `video`

这说明这套数据不只是原始视频，还已经包含了：

- 标注文件
- 视频文件
- 预处理后的帧级数组

### 2.2 数据划分

在多个目录下都能看到标准划分：

- `train`
- `dev`
- `test`

这非常适合直接用于：

- 训练
- 验证
- 测试

### 2.3 视频目录结构

`video` 目录的结构为：

```text
video/
├── train/
├── dev/
└── test/
```

每个划分下面又按 `Translator` 分为：

- `A`
- `B`
- `C`
- ...
- `L`

也就是：

```text
video/train/A/train-00001.mp4
video/train/B/...
video/dev/A/...
video/test/A/...
```

我已经确认下面这个路径是真实存在的：

```text
E:\project\xiaozhi\xiaozhi-esp32-2.2.2\xiaozhi-esp32-2.2.2\ai-models\data\CSL-500\CE-CSL\CE-CSL\video\train\A\train-00001.mp4
```

这说明：

- `Number` 字段负责样本主键
- `Translator` 字段负责子目录
- 视频路径可由 `split + Translator + Number + .mp4` 直接拼接得到

### 2.4 预处理帧目录

`frames/train` 和 `extracted_frames/train` 中都存在大量 `.npy` 文件，例如：

- `train-00001.npy`
- `train-00002.npy`

我抽查了一个样本：

- 文件：`frames/train/train-00001.npy`
- 形状：`(3, 48, 128, 128)`
- 数据类型：`uint8`

这说明这些 `.npy` 文件很可能是：

- 已经裁剪/采样后的固定长度视频张量
- 采用 `C, T, H, W` 格式

也就是说，`CE-CSL` 不只是“原始视频数据集”，还是一套带有预处理缓存的数据集。

### 2.5 标注目录

`label` 和 `label_cleaned` 中都包含：

- `train.csv`
- `dev.csv`
- `test.csv`

其中更建议你优先使用：

- `label_cleaned`

因为它比原始 `label` 多出了清洗后的 gloss 字段。

## 3. CE-CSL 标注格式分析

### 3.1 原始标注字段

`label/train.csv` 的表头为：

```text
Number,Translator,Chinese Sentences,Gloss,Note
```

### 3.2 清洗后标注字段

`label_cleaned/train.csv` 的表头为：

```text
Number,Translator,Chinese Sentences,Gloss,Note,Gloss_Original,Gloss_Clean
```

这是非常理想的字段设计，因为它同时提供了：

1. 样本编号
2. 说话人/翻译者信息
3. 中文句子
4. Gloss
5. 清洗前 gloss
6. 清洗后 gloss

### 3.3 样本示例

我抽取的一条样本如下：

- `Number`: `train-00001`
- `Translator`: `A`
- `Chinese Sentences`: `2023年高考到了。`
- `Gloss`: `2/0/2/3/高/考/时间/到`
- `Gloss_Original`: `2/0/2/3/高/考/时间/到/。`
- `Gloss_Clean`: `2/0/2/3/高/考/时间/到`

这说明这套数据对于 Uni-Sign 非常有价值，因为它已经天然提供了：

- 自然语言句子
- gloss 序列

也就是说，它既能支持：

- `SLT`：手语翻译
- `CSLR`：连续手语识别

### 3.4 样本数量

根据 `label_cleaned` 统计结果：

- `train`: 4973 条
- `dev`: 515 条
- `test`: 500 条

总计约：

- `5988` 条样本

这是一个比较适合：

- 微调
- 消融实验
- 中文连续手语论文实验

的数据规模。

## 4. CE-CSL 与 Uni-Sign 当前数据接口的差异

Uni-Sign 当前主要支持两类数据组织方式：

### 4.1 第一类：压缩标注字典

例如：

- `CSL_Daily`
- `WLASL`
- `How2Sign`
- `OpenASL`

这些数据集通过：

- `gzip + pickle`

加载成 Python 字典。

### 4.2 第二类：JSON 标注

例如：

- `CSL_News`

### 4.3 CE-CSL 的特点

`CE-CSL` 不属于以上两类，而是：

- 结构化 CSV 标注
- 明确的视频目录分层
- 附带预处理 `.npy`

所以它接入 Uni-Sign 最自然的方式不是硬改已有 `CSL_Daily` 标注文件，而是新增一套：

- `CE-CSL` 专用数据集加载逻辑

## 5. CE-CSL 接入 Uni-Sign 的推荐方案

## 5.1 总体原则

建议采用“最小侵入式改造”：

1. 不破坏现有 `CSL_Daily / WLASL / How2Sign / OpenASL` 流程
2. 新增 `CE-CSL` 数据集分支
3. 先跑通 `pose-only`
4. 再决定是否接 `RGB+Pose`

这是最稳的方案。

## 5.2 第一阶段：先做 Pose-only 接入

这是最推荐的起点。

因为 Uni-Sign 当前最稳定的数据主干仍然是：

- 视频 -> Pose -> 模型

所以第一阶段建议：

1. 先为 `CE-CSL` 批量提取 pose
2. 为 `CE-CSL` 生成 Uni-Sign 兼容标注
3. 在 `config.py` 和 `datasets.py` 中注册新数据集
4. 先跑 `SLT` 或 `CSLR` 的 pose-only 实验

## 5.3 第二阶段：再做 RGB+Pose 接入

如果第一阶段跑通了，再接：

- 视频路径
- 手部 RGB 裁剪
- 多模态融合

这样风险更小。

## 6. 具体代码改造方案

下面是我建议的最实用改造路径。

## 6.1 改造点一：在 `config.py` 注册 `CE-CSL`

需要新增：

1. `train_label_paths["CE_CSL"]`
2. `dev_label_paths["CE_CSL"]`
3. `test_label_paths["CE_CSL"]`
4. `rgb_dirs["CE_CSL"]`
5. `pose_dirs["CE_CSL"]`

推荐形式不是直接指向原始 CSV，而是指向你转换后的 Uni-Sign 兼容标注文件。

例如后续建议新增：

```text
data/CE_CSL/labels.train
data/CE_CSL/labels.dev
data/CE_CSL/labels.test
dataset/CE_CSL/video
dataset/CE_CSL/pose_format
```

## 6.2 改造点二：扩展 `utils.get_args_parser()`

在 [utils.py](E:\project\xiaozhi\Uni-Sign\utils.py) 中：

- `--dataset`

当前候选列表里没有 `CE_CSL`。

需要把它加入：

```python
choices=['CSL_News', 'CSL_Daily', 'WLASL', 'How2Sign', 'OpenASL', 'CE_CSL']
```

## 6.3 改造点三：新增 `CE_CSL` 的标注转换脚本

这是最关键的一步。

建议新建一个脚本，例如：

```text
script/prepare_cecsl_labels.py
```

功能是把：

- `label_cleaned/train.csv`
- `label_cleaned/dev.csv`
- `label_cleaned/test.csv`

转换成 Uni-Sign 当前能直接读取的压缩标签格式。

转换后的每条样本建议结构与 `S2T_Dataset` 当前读取方式对齐：

```python
{
  "sample_id": {
    "name": "train-00001",
    "text": "2023年高考到了。",
    "gloss": ["2", "0", "2", "3", "高", "考", "时间", "到"],
    "video_path": "train/A/train-00001.mp4"
  }
}
```

这里最关键的是：

- `name`：对应样本编号
- `text`：取 `Chinese Sentences`
- `gloss`：建议优先取 `Gloss_Clean` 并按 `/` 切分
- `video_path`：拼成 `train/A/train-00001.mp4`

然后再用：

- `gzip`
- `pickle`

保存成 Uni-Sign 兼容格式。

## 6.4 改造点四：给 `CE-CSL` 批量提取 pose

Uni-Sign 当前主流程依赖 `.pkl` pose 文件。

因此你需要对 `CE-CSL` 视频做批量姿态提取。

建议输出目录结构与视频路径保持一致，例如：

```text
dataset/CE_CSL/pose_format/train/A/train-00001.pkl
dataset/CE_CSL/pose_format/dev/A/...
dataset/CE_CSL/pose_format/test/A/...
```

这样你就能复用现有 `S2T_Dataset.load_pose()` 的路径拼接思路。

## 6.5 改造点五：在 `datasets.py` 中新增 `CE_CSL` 分支

最稳的方案是复用 `S2T_Dataset` 的逻辑，而不是重写一整套。

建议在 `S2T_Dataset.__init__()` 中新增：

```python
elif self.args.dataset == "CE_CSL":
    self.pose_dir = os.path.join(pose_dirs[args.dataset], phase)
    self.rgb_dir = os.path.join(rgb_dirs[args.dataset], phase)
```

前提是：

- 你的 `video_path` 只写相对路径 `A/train-00001.mp4`

或者统一成：

- `train/A/train-00001.mp4`

然后你要让 `rgb_dir` 的定义和 `video_path` 格式保持一致。

### 推荐两种做法

#### 做法 A：`rgb_dir` 指向 `dataset/CE_CSL/video`

那么 `video_path` 写：

```text
train/A/train-00001.mp4
```

#### 做法 B：`rgb_dir` 指向 `dataset/CE_CSL/video/train`

那么 `video_path` 写：

```text
A/train-00001.mp4
```

我更推荐做法 A，因为它更统一。

## 6.6 改造点六：先不使用 `frames/*.npy`

虽然 `CE-CSL` 自带 `.npy` 预处理帧，但当前 Uni-Sign 现有主流程更依赖：

- 原视频 `.mp4`
- pose `.pkl`

因此第一阶段不建议强行把 `.npy` 接入主流程。

更好的策略是：

1. 先忽略 `.npy`
2. 用原视频和提取后的 pose 接入 Uni-Sign
3. 后续如果要优化速度，再考虑把 `.npy` 用作缓存输入

## 7. 最推荐的数据接入路线

### 第一步：标注转换

把 `label_cleaned/*.csv` 转成 Uni-Sign 可读的 `labels.train/dev/test`

### 第二步：pose 提取

对 `video/train`、`video/dev`、`video/test` 批量提取 `.pkl`

### 第三步：配置注册

在：

- [config.py](E:\project\xiaozhi\Uni-Sign\config.py)
- [utils.py](E:\project\xiaozhi\Uni-Sign\utils.py)

中注册 `CE_CSL`

### 第四步：数据集加载分支

在：

- [datasets.py](E:\project\xiaozhi\Uni-Sign\datasets.py)

中增加 `CE_CSL` 的路径处理逻辑

### 第五步：先跑 pose-only

先验证：

- `SLT`

或：

- `CSLR`

### 第六步：再接 RGB+Pose

在 pose-only 稳定后再做手部 RGB 融合。

## 8. 论文价值判断

`CE-CSL` 对你后续论文非常有价值，原因有三点：

### 8.1 它是中文连续手语数据

这和你当前的应用目标高度一致：

- 中文场景
- 连续手语
- 公共服务沟通

### 8.2 它同时带句子和 gloss

这意味着你可以同时研究：

1. `CSLR`
2. `SLT`

甚至可以探索：

- gloss 中间表示
- gloss-free 翻译
- 中文连续手语统一建模

### 8.3 它适合做你自己的“场景化升级”

你完全可以把 CE-CSL 作为：

- 论文主实验数据集
- 中文鲁棒性验证数据集
- 自建应用适配扩展底座

## 9. 结论

基于当前分析，`CE-CSL` 是一套结构清晰、标注字段完整、非常适合接入 Uni-Sign 的中文连续手语数据集。

它的优点在于：

1. 有标准 `train/dev/test`
2. 有 `Chinese Sentences`
3. 有 `Gloss_Clean`
4. 视频路径规则清晰
5. 还有 `.npy` 预处理缓存

因此，最推荐的 Uni-Sign 适配路线是：

1. 先做 CSV -> Uni-Sign 标注转换
2. 再做视频 -> pose 批量提取
3. 再注册 `CE_CSL`
4. 先跑 pose-only
5. 最后扩展到 RGB+Pose

如果你下一步愿意，我可以继续直接帮你做两件更实操的事：

1. 先给你写 `CE-CSL` 标注转换脚本设计稿
2. 再给你写 Uni-Sign 代码里需要改的具体位置清单

## 10. 已经完成的代码改造

为了让 `CE-CSL` 能接入 Uni-Sign，我已经先完成了第一批基础改造。

### 10.1 已修改文件

#### 1. [config.py](E:\project\xiaozhi\Uni-Sign\config.py)

已新增：

- `CE_CSL` 的 `train/dev/test` 标签路径
- `CE_CSL` 的 `rgb_dirs`
- `CE_CSL` 的 `pose_dirs`
- 环境变量 `CE_CSL_ROOT` 支持

默认 `CE_CSL_ROOT` 已写成你当前本机的数据路径。

#### 2. [utils.py](E:\project\xiaozhi\Uni-Sign\utils.py)

已把：

- `CE_CSL`

加入 `--dataset` 的可选列表。

#### 3. [datasets.py](E:\project\xiaozhi\Uni-Sign\datasets.py)

已为 `S2T_Dataset` 增加：

- `CE_CSL` 的路径分支

当前逻辑假设：

- 标注中的 `video_path` 形如 `train/A/train-00001.mp4`
- `rgb_dir` 指向 `CE-CSL/video`
- `pose_dir` 指向 `CE-CSL/pose_format`

#### 4. [prepare_cecsl_labels.py](E:\project\xiaozhi\Uni-Sign\script\prepare_cecsl_labels.py)

已新增一个标注转换脚本，用于把：

- `label_cleaned/train.csv`
- `label_cleaned/dev.csv`
- `label_cleaned/test.csv`

转换成 Uni-Sign 当前可直接读取的：

- `data/CE_CSL/labels.train`
- `data/CE_CSL/labels.dev`
- `data/CE_CSL/labels.test`

## 11. 你下一步怎么操作

下面这部分是最重要的实操顺序。

### 第一步：先生成 CE-CSL 标签文件

在 Uni-Sign 根目录执行：

```powershell
python .\script\prepare_cecsl_labels.py
```

如果你后面换了 CE-CSL 路径，也可以显式指定：

```powershell
python .\script\prepare_cecsl_labels.py --cecsl_root "E:\your\CE-CSL\CE-CSL"
```

运行成功后，应生成：

```text
data\CE_CSL\labels.train
data\CE_CSL\labels.dev
data\CE_CSL\labels.test
```

### 第二步：为 CE-CSL 视频批量提取 pose

当前 Uni-Sign 主流程仍然依赖 `.pkl` pose 文件，因此你需要为 CE-CSL 生成：

```text
CE-CSL\pose_format\train\A\train-00001.pkl
CE-CSL\pose_format\dev\A\...
CE-CSL\pose_format\test\A\...
```

注意：

- 当前 `demo/pose_extraction.py` 只支持“单目录扁平扫描”
- 而 `CE-CSL` 视频是多层目录结构

因此这里最推荐的办法不是直接改现有主逻辑，而是后续写一个：

- `prepare_cecsl_pose.py`

专门递归处理：

- `video/train`
- `video/dev`
- `video/test`

并保持子目录结构不变。

### 第三步：确认目录关系

在当前代码假设下，目录应该满足：

```text
CE-CSL/
├── video/
│   ├── train/A/train-00001.mp4
│   ├── dev/A/...
│   └── test/A/...
└── pose_format/
    ├── train/A/train-00001.pkl
    ├── dev/A/...
    └── test/A/...
```

### 第四步：先跑 pose-only

在 pose 文件准备好之后，优先尝试：

- `SLT`

或：

- `CSLR`

的 pose-only 流程，不要一上来就做 RGB+Pose。

### 第五步：再决定是否扩展 RGB+Pose

等你确认：

1. 标签转换没问题
2. pose 读取没问题
3. 数据路径没问题

之后，再考虑完整的 RGB+Pose 训练与推理。

## 12. 当前还没做但下一步最值得做的代码改造

### 12.1 最推荐新增的脚本

下一步最值得继续补的是：

1. `script/prepare_cecsl_pose.py`
2. `script/check_cecsl_dataset.py`

作用分别是：

- 自动提取并生成 `pose_format`
- 自动检查视频、pose、标签是否一一对应

这两个脚本现在已经补充完成。

#### [prepare_cecsl_pose.py](E:\project\xiaozhi\Uni-Sign\script\prepare_cecsl_pose.py)

作用：

- 递归扫描 `CE-CSL/video/train|dev|test`
- 保持原有子目录结构
- 生成：

```text
CE-CSL/pose_format/train/A/train-00001.pkl
```

推荐运行方式：

```powershell
python .\script\prepare_cecsl_pose.py
```

如果你后续改了 CE-CSL 路径，也可以：

```powershell
python .\script\prepare_cecsl_pose.py --cecsl_root "E:\your\CE-CSL\CE-CSL"
```

#### [check_cecsl_dataset.py](E:\project\xiaozhi\Uni-Sign\script\check_cecsl_dataset.py)

作用：

- 检查转换后的 `labels.train/dev/test`
- 检查视频文件是否存在
- 检查 pose 文件是否存在
- 可选检查 pose 文件内容是否完整

推荐运行方式：

```powershell
python .\script\check_cecsl_dataset.py
```

如果你想额外检查 `.pkl` 内容是否损坏：

```powershell
python .\script\check_cecsl_dataset.py --check_pose_content
```

### 12.2 后续最推荐的实验起点

对于 CE-CSL，最推荐的起点是：

1. 先做 `SLT`
2. 再做 `CSLR`

原因是：

- 它天然带中文句子
- 也带 gloss
- 两类任务都可以做

但从项目落地和比赛展示角度，`SLT` 更容易展示价值。

## 13. 当前最推荐的执行顺序

你现在可以按下面顺序继续推进：

1. 运行 [prepare_cecsl_labels.py](E:\project\xiaozhi\Uni-Sign\script\prepare_cecsl_labels.py)
2. 运行 [prepare_cecsl_pose.py](E:\project\xiaozhi\Uni-Sign\script\prepare_cecsl_pose.py)
3. 运行 [check_cecsl_dataset.py](E:\project\xiaozhi\Uni-Sign\script\check_cecsl_dataset.py)
4. 确认 `labels + video + pose` 都已对齐
5. 再尝试在 Uni-Sign 中以 `CE_CSL` 作为 `--dataset` 运行
---

## 13. 最新执行建议：把 CE-CSL 真正推到可训练状态

前面完成的是“最小可运行链路”，也就是：

- `labels`
- `pose`
- `dataset loading`
- `minimal forward`

这说明项目已经能读入 `CE-CSL`，但还不代表对 `CE-CSL` 有真实可用精度。

当前你看到预测偏差较大，本质原因是：

- 现在用的是现成 Uni-Sign 权重
- 不是在 `CE-CSL` 上微调出来的权重

因此后面的主线应该非常明确：

1. 继续补更多 `CE-CSL pose_format`
2. 切到 `CE-CSL` 上做 `pose-only + SLT` 微调
3. 训练出 `CE-CSL` 自己的 `best_checkpoint.pth`

### 13.1 继续补 pose 的建议

如果是 Windows 本机，只建议继续生成一小批到中等规模样本，用来：

- 看效果
- 做答辩演示
- 验证训练链路前的数据准备

推荐分阶段执行：

```powershell
python .\script\prepare_cecsl_pose.py --splits train --max_videos_per_split 20 --max_workers 4
python .\script\check_cecsl_dataset.py

python .\script\prepare_cecsl_pose.py --splits train --max_videos_per_split 200 --max_workers 4
python .\script\check_cecsl_dataset.py
```

如果要做全量：

```bash
python ./script/prepare_cecsl_pose.py --splits train dev test --max_workers 4
python ./script/check_cecsl_dataset.py --check_pose_content
```

更推荐放在 Ubuntu 服务器上完成。

### 13.2 最推荐的第一版微调任务

第一版不要贪多，直接锁定：

- `dataset = CE_CSL`
- `task = SLT`
- `pose-only`

原因：

- 你当前已经验证 `pose` 链路是通的
- `RGB+Pose` 会增加额外复杂度
- 先做出第一版可提升结果最重要

### 13.3 微调起点

建议直接以你已经下载好的现成权重作为初始化：

```text
out/stage3_finetuning/best_checkpoint.pth
```

然后在此基础上继续适配 `CE-CSL`。

### 13.4 Ubuntu 上的第一版训练命令模板

```bash
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir out/cecsl_stage3_pose_only \
  --finetune out/stage3_finetuning/best_checkpoint.pth \
  --dataset CE_CSL \
  --task SLT
```

### 13.5 第一轮训练目标

第一轮训练不追求最好指标，只追求三件事：

1. 训练过程稳定
2. 能成功保存 `best_checkpoint.pth`
3. 预测结果开始贴近 `CE-CSL` 真值

### 13.6 训练后的最小验证

训练完成后，优先做抽样前向验证：

```powershell
python .\script\test_cecsl_forward.py --split train --sample_index 0 --device cuda --dtype fp32 --checkpoint .\out\cecsl_stage3_pose_only\best_checkpoint.pth
python .\script\test_cecsl_forward.py --split train --sample_index 1 --device cuda --dtype fp32 --checkpoint .\out\cecsl_stage3_pose_only\best_checkpoint.pth
```

如果预测内容明显更接近：

- `2023年高考到了。`
- `一个小时的时间才使她明白我的意思。`

就说明 `CE-CSL` 微调方向是有效的。

### 13.7 后续升级路线

等第一版 `pose-only + SLT` 稳定之后，再继续：

1. dev/test 正式评估
2. 更多样本抽检
3. `RGB+Pose`
4. 消融实验
5. 鲁棒性实验
6. 论文结果表和图表整理
