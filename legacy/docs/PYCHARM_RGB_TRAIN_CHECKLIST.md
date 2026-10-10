# PyCharm 终端 RGB 训练清单

只看这个文件，只在 PyCharm Terminal 里执行。

## 1. 先下载这些

必须有：

- `pretrained_weight/mt5-base`
- `out/stage3_finetuning/best_checkpoint.pth`
- `data/CE_CSL/labels.train`
- `data/CE_CSL/labels.dev`
- `data/CE_CSL/labels.test`
- `data/CE_CSL/CE-CSL/video`
- `data/CE_CSL/CE-CSL/pose_format`

强烈建议再下载：

- `pretrained_weight/efficientnet_b0_rwightman-7f5810bc.pth`

没有这一个也能跑，但 RGB backbone 会随机初始化。

## 2. PyCharm 里先做体检

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action doctor
```

## 3. RGB smoke

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action rgb-smoke -Device cuda -DType bf16 -FreezeMt5
```

## 4. RGB round1

冻结 `mT5`，先让 RGB 分支接上来：

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action rgb-train -Device cuda -DType bf16 -FreezeMt5
```

输出目录：

```text
out/cecsl_rgb_support_round1
```

## 5. round1 评估

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action rgb-eval-dev -Checkpoint .\out\cecsl_rgb_support_round1\best_checkpoint.pth -Device cpu -DType fp32 -SampleCount 20
```

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action rgb-eval-test -Checkpoint .\out\cecsl_rgb_support_round1\best_checkpoint.pth -Device cpu -DType fp32 -SampleCount 20
```

## 6. RGB round2 unfreeze

如果 round1 比 pose-only 明显更稳，就继续：

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action rgb-train-unfreeze -Checkpoint .\out\cecsl_rgb_support_round1\best_checkpoint.pth -Device cuda -DType bf16
```

输出目录：

```text
out/cecsl_rgb_support_round2_unfreeze
```

## 7. round2 评估

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action rgb-eval-dev -Checkpoint .\out\cecsl_rgb_support_round2_unfreeze\best_checkpoint.pth -Device cpu -DType fp32 -SampleCount 20
```

```powershell
powershell -ExecutionPolicy Bypass -File .\script\start_cecsl_windows.ps1 -Action rgb-eval-test -Checkpoint .\out\cecsl_rgb_support_round2_unfreeze\best_checkpoint.pth -Device cpu -DType fp32 -SampleCount 20
```

## 8. Ubuntu 后面继续什么

如果你 Ubuntu 还在训练中，后面只接这一条：

```text
stage3 基座
-> RGB round1 freeze
-> RGB round2 unfreeze
-> dev/test 抽样评估
-> 小步长 refine
```
