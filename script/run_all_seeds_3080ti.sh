#!/usr/bin/env bash
# ============================================================================
# 3080Ti 全量种子训练 — 一次性跑完，不断线
# ============================================================================
# 用法:
#   chmod +x script/run_all_seeds_3080ti.sh
#   nohup bash script/run_all_seeds_3080ti.sh > logs/seed_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   或:  tmux new -s seed_train \; bash script/run_all_seeds_3080ti.sh
#
# 断线后重连查看:  tail -f logs/seed_train_*.log
# 恢复进度:       bash script/run_all_seeds_3080ti.sh --start-from seed789_full_model
# ============================================================================

set -euo pipefail

# ── 路径配置 (按你的 3080Ti Linux 实际路径修改) ──────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_EXE="${HOME}/.conda/envs/torch13/bin/python"
STORAGE_OUT="/mnt/E/project/xiaozhi/Uni-Sign-storage/out"
# 如果 storage 路径不同，改上面这行

TRAIN_SCRIPT="${PROJECT_ROOT}/local_sign_tools/finetune_cecsl_rgb_support.py"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${STORAGE_OUT}"

# ── 3080Ti 20GB 优化参数 ──────────────────────────────────────
BASE_ARGS=(
    --epochs 2 --stage2_epochs 2
    --stage2_lr 1e-7 --stage2_mt5_lr_scale 0.1
    --batch_size 8 --gradient_accumulation_steps 1
    --weight_decay 1e-4 --max_length 256 --hidden_dim 256
    --eval_sample_limit 100 --eval_split test
    --dtype bf16 --num_workers 4
)

# ── 实验定义 ──────────────────────────────────────────────────
# 全部 7 个未完成种子: 456, 789, 101, 202, 303, 404, 505
# 每种子 3 变体: pose_only → naive_rgb → full_model
# 共 21 个实验

SEEDS=(456 789 101 202 303 404 505)
VARIANTS=("pose_only" "naive_rgb" "full_model")
declare -A EXTRA_ARGS

EXTRA_ARGS["pose_only"]="--no_rgb_support"
EXTRA_ARGS["naive_rgb"]="--train_rgb_frame_sampling confidence_topk --eval_rgb_frame_sampling confidence_topk"
EXTRA_ARGS["full_model"]="--train_rgb_frame_sampling confidence_topk --eval_rgb_frame_sampling confidence_topk --fusion_use_uncertainty_gate --fusion_meta_dim 3"

# ── 解析参数 ──────────────────────────────────────────────────
START_FROM=""
RESUME=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-from) START_FROM="$2"; shift 2 ;;
        --resume)     RESUME=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── 开始 ──────────────────────────────────────────────────────
RUN_LOG="${LOG_DIR}/seed_train_results_$(date +%Y%m%d_%H%M%S).jsonl"
TOTAL=$(( ${#SEEDS[@]} * ${#VARIANTS[@]} ))
DONE=0
SKIP=true
[[ -z "${START_FROM}" ]] && SKIP=false

echo "============================================================================"
echo "  3080Ti 全量种子训练"
echo "  项目: ${PROJECT_ROOT}"
echo "  Python: ${PYTHON_EXE}"
echo "  Storage: ${STORAGE_OUT}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Variants: ${VARIANTS[*]}"
echo "  总数: ${TOTAL} 个实验"
echo "  开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志: ${RUN_LOG}"
echo "============================================================================"

for seed in "${SEEDS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        LABEL="seed${seed}_${variant}"

        # ── skip/resume 逻辑 ──
        if ${SKIP}; then
            if [[ "${LABEL}" != "${START_FROM}" ]]; then
                echo "[SKIP] ${LABEL} (before start-from)"
                continue
            else
                SKIP=false
                echo "[RESUME] 从 ${LABEL} 开始"
            fi
        fi

        # ── 检查是否已完成（跳过已有 checkpoint 的实验）──
        OUT_DIR="${STORAGE_OUT}/${LABEL}"
        if [[ -f "${OUT_DIR}/final_checkpoint.pth" ]] && ! ${RESUME}; then
            echo "[SKIP] ${LABEL} — 已有 final_checkpoint.pth，跳过"
            DONE=$((DONE + 1))
            continue
        fi

        # ── 构建命令 ──
        read -ra EXTRA <<< "${EXTRA_ARGS[${variant}]}"
        CMD=("${PYTHON_EXE}" "${TRAIN_SCRIPT}"
             --output_dir "${OUT_DIR}"
             "${BASE_ARGS[@]}"
             "${EXTRA[@]}"
             --seed "${seed}"
        )

        echo ""
        echo "======================================================================"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: ${LABEL}"
        echo "  dir: ${OUT_DIR}"
        echo "  cmd: ${CMD[*]}"
        echo "======================================================================"

        T0=$(date +%s)

        # ── 运行训练 ──
        PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1
        EXIT_CODE=$?

        ELAPSED=$(( $(date +%s) - T0 ))
        HOURS=$(echo "scale=1; ${ELAPSED}/3600" | bc)

        if [[ ${EXIT_CODE} -eq 0 ]]; then
            STATUS="OK"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: ${LABEL} → ${STATUS} (${HOURS}h)"
        else
            STATUS="FAILED"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL: ${LABEL} (exit=${EXIT_CODE}, ${HOURS}h)"
        fi

        # ── 记录结果 ──
        echo "{\"label\":\"${LABEL}\",\"seed\":${seed},\"variant\":\"${variant}\",\"status\":\"${STATUS}\",\"elapsed_h\":${HOURS},\"timestamp\":\"$(date '+%Y-%m-%d %H:%M:%S')\"}" >> "${RUN_LOG}"

        DONE=$((DONE + 1))
        echo "[PROGRESS] ${DONE}/${TOTAL} 完成"
    done
done

# ── 汇总 ──────────────────────────────────────────────────────
echo ""
echo "============================================================================"
echo "  全部完成!"
echo "  结束: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  结果: ${RUN_LOG}"
echo "============================================================================"
echo ""
echo "  Seed        Variant       Status     Time(h)"
echo "  ----------  ------------  ---------  -------"
while IFS= read -r line; do
    python3 -c "
import json,sys
r=json.loads('${line}')
print(f\"  {r['label']:<12} {r['variant']:<12} {r['status']:<10} {r['elapsed_h']}\")
" 2>/dev/null || echo "  ${line}"
done < "${RUN_LOG}"

# 统计失败数
FAIL_COUNT=$(grep -c '"FAILED"' "${RUN_LOG}" 2>/dev/null || echo 0)
OK_COUNT=$(grep -c '"OK"' "${RUN_LOG}" 2>/dev/null || echo 0)
echo ""
echo "  成功: ${OK_COUNT} / 失败: ${FAIL_COUNT} / 总计: $((OK_COUNT + FAIL_COUNT))"
if [[ ${FAIL_COUNT} -gt 0 ]]; then
    echo "  ⚠ 有失败实验，查看日志: grep FAILED ${RUN_LOG}"
fi
