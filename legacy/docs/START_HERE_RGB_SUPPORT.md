# CE-CSL RGB-Support 主线

这份文件替代 `pose-only` 主线。

如果你已经确认:

- `pose-only` 长时间不收敛
- 文本输出总是“有点像，但不准”
- 你想把项目推进到更接近实际可用的方向

那么从现在开始，建议主线改成:

```text
CE-CSL + RGB-support
```

## 1. 当前结论

截至 **2026-04-06**，我已经替你做了本机核查，结论如下:

### 已确认可用

- `CE-CSL` 标签可用
- `CE-CSL` 视频可用
- `CE-CSL` pose 可用
- `stage3_finetuning/best_checkpoint.pth` 可作为 RGB-support 初始化基座
- `local_sign_tools/finetune_cecsl_rgb_support.py` 能实际跑通最小训练
- `strict=False` 加载已经生效，RGB 分支缺失键不会报错中断

### 已实际验证到的现象

我已经跑过一轮最小 RGB-support 训练:

```text
checkpoint: out/stage3_finetuning/best_checkpoint.pth
train_sample_limit: 2
eval_sample_limit: 2
epochs: 1
device: cpu
dtype: fp32
freeze_mt5: True
```

得到的关键信息:

- `RGB-support` 训练链路可启动
- `missing_keys: 336` 是正常现象
  - 因为原始 `stage3_finetuning` checkpoint 没有 RGB 分支参数
- 当前缺少:
  - `pretrained_weight/efficientnet_b0_rwightman-7f5810bc.pth`
- 所以现在 RGB backbone 是:
  - **随机初始化**

这说明两件事:

1. 你现在已经可以正式转 RGB-support
2. 但当前视觉分支起点还不够好，未来效果上限会受影响

## 2. 为什么 RGB-support 比 pose-only 更值得继续

`pose-only` 的问题不是“完全没意义”，而是它更像:

- 链路验证方案
- 轻量基线方案
- 语义粗粒度方案

它的天然上限是:

- 手形细节损失大
- 手部纹理完全缺失
- 一些相似动作只靠骨架很难分
- 连续手语里很多细粒度差异会被抹掉

而 `RGB-support` 至少补上了:

- 手部局部外观
- 部分指形细节
- 左右手局部图像信息
- pose 对齐后的视觉支持特征

所以如果你想继续追求:

```text
更准的中文句子
更少的模板化泛化
更接近真实应用
```

那下一条主线就应该是 RGB-support。

## 3. 这条线的真实预期

先说实话:

```text
RGB-support 不等于立刻生产级。
```

但它比 pose-only 更有希望把你从:

- “语义大概相关”

推到:

- “很多句子开始明显更接近原意”

### 你可以期待的改善

- 一些句子不再只输出泛化模板
- 手部相关动作更容易分开
- 语义偏差有机会明显缩小
- 对 CE-CSL 这种真实中文连续手语更有提升空间

### 你暂时不要期待的事

- 直接达到商业部署质量
- 在普通笔记本上轻松跑大规模长周期训练
- 一轮训练就彻底解决语义漂移
- 立刻变成低延迟实时系统

## 4. 现在最关键的两个现实问题

### 问题 1: 你还没有 EfficientNet 本地预训练权重

当前仓库里只有:

- `pretrained_weight/mt5-base`

没有:

- `pretrained_weight/efficientnet_b0_rwightman-7f5810bc.pth`

这会导致:

- RGB backbone 随机初始化
- 训练更慢
- 更难稳定
- 更容易学不到强视觉信息

### 问题 2: 你现在没有真正训练完成的 RGB-support 最佳模型

仓库虽然有目录:

- `out/cecsl_rgb_support_round4_refine_win`

但当前里面没有实际可评估的 `best_checkpoint.pth`。

所以现在正确状态不是“我们已经有 RGB 最佳模型”，而是:

```text
RGB-support 代码和数据已通，但正式训练还没闭环。
```

## 5. 正确的 RGB-support 完整训练方法

建议你按下面四段来，而不是一上来就长时间盲跑。

### 阶段 A: 烟雾验证

目标:

- 确认 RGB-support 真能训
- 确认不会在 DataLoader / 显存 / 权重加载处崩掉

命令:

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action rgb-smoke -Device cuda -DType bf16 -FreezeMt5
```

### 阶段 B: 第一轮正式训练

目标:

- 从 `stage3_finetuning` 权重接出 RGB 分支
- 先冻结 `mT5`
- 让 RGB 分支和 pose 分支先学会配合

建议设置:

- `checkpoint = out/stage3_finetuning/best_checkpoint.pth`
- `freeze_mt5 = True`
- `epochs = 2~3`
- `lr = 1e-6`
- `batch_size = 1`
- `gradient_accumulation_steps = 4 或 8`

命令:

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action rgb-train -Device cuda -DType bf16 -FreezeMt5
```

默认输出目录:

```text
out/cecsl_rgb_support_round1
```

### 阶段 C: 解冻语言模型继续训练

目标:

- 不再只让视觉分支适配
- 开始让语言端共同调整
- 争取把“语义相关但不准”往“语义更准确”推进

前提:

- 阶段 B 已经拿到自己的 `best_checkpoint.pth`

建议设置:

- 从 `out/cecsl_rgb_support_round1/best_checkpoint.pth` 继续
- 去掉 `freeze_mt5`
- 更小学习率，比如 `3e-7 ~ 1e-6`
- 再训 `2~4` 个 epoch

这一步我没有做成单独脚本入口，是因为你需要先拿到 round1 checkpoint。

### 阶段 D: 抽样评估和小步 refine

目标:

- 不靠 loss 自我安慰
- 用抽样结果判断方向对不对

命令:

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action rgb-eval-dev -Checkpoint .\out\cecsl_rgb_support_round1\best_checkpoint.pth -Device cpu -DType fp32
```

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action rgb-eval-test -Checkpoint .\out\cecsl_rgb_support_round1\best_checkpoint.pth -Device cpu -DType fp32
```

如果抽样明显更贴近原文，再继续下一轮 refine。

## 6. 你在 PyCharm Terminal 里的推荐工作流

以后在 PyCharm 终端里只按这个顺序走:

1. 先做资源体检
2. 再做 RGB-support 烟雾训练
3. 再开 round1 正式训练
4. 保存 best checkpoint
5. 用 dev/test 抽样评估
6. 再决定是否 unfreeze mt5

## 7. 你该怎么判断“RGB-support 值不值得继续”

如果出现下面这些改善，就说明方向是对的:

- 模板句减少
- 原句关键信息保留更多
- 主语/动作/时间信息错误减少
- 抽样主观质量明显好于 pose-only

如果训练后仍然是:

- 看起来比 pose-only 只好一点点
- 或者几乎没变化

那优先怀疑这几个原因:

1. RGB backbone 还是随机初始化
2. 训练轮次太短
3. `mT5` 一直冻结，语义层没适配
4. 采样和评估样本太少，误判了方向

## 8. 未来效果怎么判断

### 短期效果

短期最现实的目标不是“生产级”，而是:

```text
把 CE-CSL 的输出从“相关但飘”推到“明显更接近原句”。
```

### 中期效果

如果你补上 EfficientNet 本地预训练权重，再做:

- round1 冻结训练
- round2 小学习率解冻训练
- 连续抽样评估

那么这条线有机会达到:

- 比 pose-only 明显更可用
- 更适合作为视频离线识别主模型
- 更适合继续往在线系统推进

### 长期效果

真正接近生产，还需要再补:

- 更稳定的视频切段
- 在线滑动窗口
- 预测去抖
- 后处理与纠错
- 产品层交互

所以正确期待是:

```text
RGB-support 是通向生产级的必要一步，但不是最后一步。
```

## 9. 现在只记住三句话

```text
pose-only 先停。
RGB-support 现在可以正式开跑。
真正的上限还取决于视觉 backbone 权重和后续 unfreeze 微调。
```
