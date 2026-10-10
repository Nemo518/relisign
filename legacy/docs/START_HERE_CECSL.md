# Uni-Sign CE-CSL 重整入口

这份文件是当前项目的唯一建议入口。

如果你现在的目标是:

- 先把项目跑起来
- 先确认 `CE-CSL` 链路是通的
- 再逐步逼近“可实际使用”的智能手语识别

那就先只看这份文件，先不要同时看 `docs/` 里的旧计划、旧日程、旧分轮记录。

## 1. 先说结论

截至 **2026-04-06**，这个仓库的状态可以一句话概括成:

```text
链路已经打通，但当前最好结果还不是生产级。
```

我已经在本机做过最小验证，结果如下:

- `mt5-base` 权重存在且可加载
- 官方 `out/stage3_finetuning/best_checkpoint.pth` 存在且可加载
- `CE-CSL` 标签已就位:
  - train: `4973`
  - dev: `515`
  - test: `500`
- `CE-CSL` pose 已就位:
  - train: `4969`
  - dev: `530`
  - test: `500`
- 训练集只有 `4` 个样本缺视频和 pose，不会阻塞主线
- `out/cecsl_pose_only_round4/best_checkpoint.pth` 已能完成前向推理

但抽样结果也说明一个事实:

- 模型已经能生成“语义相关”的句子
- 但语义偏差仍然明显
- 还不能叫“生产可用”

抽样例子:

```text
GT:  10年的禁鱼区不是很长时间。
Pred: 这个工作不是很久。

GT:  2023年的高考有一千多万考生。
Pred: 这几年,高中生总是迟到的。

GT:  下雨后空气会变好。
Pred: 下雨了,天气很好。
```

所以你现在最需要的不是继续迷失在“到底该跑哪个脚本”，而是接受下面这个现实:

```text
当前 pose-only 路线已经证明“项目能跑”，但还没有证明“项目能生产”。
```

## 2. 现在只保留一条主线

从现在开始，建议只保留下面这条路线:

```text
环境确认
-> 资源体检
-> 最小前向验证
-> 最小训练烟雾测试
-> CE-CSL pose-only 正式微调
-> 抽样评估
-> 摄像头/视频演示
-> 再进入 RGB+pose 主线
```

## 3. 现在先不要碰什么

为了避免继续混乱，下面这些东西先不要作为主入口:

- `script/train_stage1.sh`
- `script/train_stage2.sh`
- `script/train_stage3.sh`
- `demo/online_inference.py`
- `docs/` 里按日期写的日程文档
- 任何还没有实际 checkpoint 的 `rgb_support` 计划目录

原因很简单:

- `.sh` 更偏官方多卡训练流程，不适合你现在的 Windows 本机整理期
- `demo/online_inference.py` 更像单视频示例，不是你的项目主控制台
- 你当前真正已经接通的是 `CE-CSL` 本地链路
- 你当前真正能跑通的模型是 `pose-only round4`

## 4. 现在应该用哪些文件

当前建议只认下面这些文件:

- `script/check_project_resources.py`
- `script/check_cecsl_dataset.py`
- `script/test_cecsl_forward.py`
- `script/eval_cecsl_samples.py`
- `local_sign_tools/train_cecsl_smoke.py`
- `local_sign_tools/finetune_cecsl_pose_only.py`
- `local_sign_tools/webcam_demo.py`
- `script/start_cecsl_windows.ps1`

## 5. 先确认你在用对环境

当前仓库里的默认 `python` 是 `Python 3.13.5`，里面没有 `torch`，不能作为训练入口。

你当前本机已经存在可用环境:

```text
C:\Users\Nemo518\.conda\envs\torch13\python.exe
```

建议以后都通过下面这个入口跑:

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action doctor
```

如果你想手动指定 Python:

```powershell
$env:UNI_SIGN_PYTHON="C:\Users\Nemo518\.conda\envs\torch13\python.exe"
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action doctor
```

## 6. 推荐执行顺序

### 第 1 步: 项目体检

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action doctor
```

目标:

- 确认 `mt5-base` 在
- 确认 checkpoint 在
- 确认 `CE-CSL` labels 在
- 确认 `pose_format` 在

### 第 2 步: 最小前向验证

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action forward -Split dev -SampleIndex 0 -Device cpu -DType fp32
```

目标:

- 证明“模型真的能读入 CE-CSL 并吐出中文句子”

### 第 3 步: 最小训练烟雾测试

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action smoke -Device cuda -DType bf16 -FreezeMt5
```

目标:

- 证明“当前机器真的能反向传播”
- 证明“不是只会推理，不会训练”

### 第 4 步: 正式微调

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action train -Device cuda -DType bf16 -FreezeMt5
```

默认会从这里继续:

```text
.\out\cecsl_pose_only_round4\best_checkpoint.pth
```

输出默认会保存到:

```text
.\out\cecsl_pose_only_round5_unfreeze
```

如果你想先继续冻结 `mT5`，可以保留 `-FreezeMt5`。

如果你想尝试更强的语义适配，再去掉 `-FreezeMt5`。

## 7. 怎么判断当前阶段算不算成功

### 当前阶段的成功标准

不是论文指标，不是生产指标，而是这三个:

- 命令能稳定跑完
- checkpoint 能稳定保存
- 抽样结果比 `round4` 更贴近原句

### 当前阶段不该强求的事

先不要强求:

- 真正的实时连续低延迟识别
- 复杂场景鲁棒性
- 生产级精度
- 纯靠 pose-only 达到高可信业务效果

## 8. 什么时候转 RGB+pose

当出现下面任意一种情况，就该停止继续堆 `pose-only` 轮次:

- 新一轮模型输出仍然大量泛化成模板句
- 语义相关但关键信息总是错
- loss 下降，但抽样文本没有实质提升
- 你已经确认数据链路和训练链路都没问题

这时就不要再“盲目 round5 / round6 / round7 pose-only”，而是转:

```text
CE-CSL + RGB-support + 更长周期微调
```

## 9. 关于“生产可用”的真实路线

如果你的目标真的是“能应用到实际生产生活中的级别”，那项目最终至少要补齐下面四层:

### 第一层: 离线模型效果

- 单视频输入
- 稳定输出较准确中文句子
- dev/test 抽样明显可用

### 第二层: 在线时序处理

- 滑动窗口
- 自动切段
- 低延迟重复推理
- 上下文去抖

### 第三层: 视觉鲁棒性

- 光照变化
- 背景变化
- 遮挡
- 不同身高和拍摄距离
- 更复杂的手部细节

### 第四层: 产品层

- 摄像头接入
- UI 展示
- 历史记录
- 错误恢复
- 延迟控制
- 监控和日志

所以不要再给自己施加一个错误压力:

```text
“只要今天再训一轮，明天就生产级了。”
```

真正正确的说法应该是:

```text
先把 CE-CSL 主线稳定跑通，再把效果从 pose-only 推到 RGB+pose，最后再做在线系统。
```

## 10. 你现在最推荐的目标

如果只看这一个星期，最合理的目标不是“直接生产级”，而是:

```text
拿到一个你自己能解释、能复现、能继续迭代的 CE-CSL 主模型。
```

这比继续在几十个脚本里迷路重要得多。

## 11. 一句话执行版

以后先只记住这 5 条:

1. 用 `torch13` 环境，不用系统默认 `python`
2. 先跑 `doctor`
3. 再跑 `forward`
4. 再跑 `smoke`
5. 然后只沿 `local_sign_tools/finetune_cecsl_pose_only.py` 往前推进

如果这一轮仍然只是“语义相关但不准”，下一步就转 `RGB+pose`，不要继续陷在 `pose-only` 里。
