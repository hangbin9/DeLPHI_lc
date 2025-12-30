#!/bin/bash
#
# Pipeline Continuation Monitor
# Watches for PeriodNet1D checkpoint completion, then runs Stage-2 + Stage-3
#

set -e

CHECKPOINT_DIR="/mnt/d/Downloads/Colab Notebooks/checkpoints/periodnet_simple_hybrid"
BEST_CHECKPOINT="$CHECKPOINT_DIR/best_model.pt"
LOG_DIR="/tmp"
WORK_DIR="/mnt/d/Downloads/Colab Notebooks"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PeriodNet V10 Pipeline Continuation Monitor"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Phase: MONITORING"
echo "Task: Wait for PeriodNet1D (Stage-1) retraining to complete"
echo ""
echo "Expected checkpoint location:"
echo "  $BEST_CHECKPOINT"
echo ""
echo "Next phases after checkpoint is ready:"
echo "  ✓ Stage-2: LightGBM training with corrected CNN features"
echo "  ✓ Stage-3: Soft voting ensemble evaluation"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Monitor for checkpoint completion
echo "Monitoring for checkpoint... (checking every 30 seconds)"
echo ""

max_wait_seconds=$((3600 * 5))  # 5 hours max wait
elapsed=0
check_interval=30

while [ ! -f "$BEST_CHECKPOINT" ]; do
    if [ $elapsed -ge $max_wait_seconds ]; then
        echo "❌ Timeout: Checkpoint not created within 5 hours"
        echo "Possible issues:"
        echo "  - Training crashed or was interrupted"
        echo "  - Checkpoint save failed"
        echo "  - Different checkpoint path than expected"
        echo ""
        echo "Debug info:"
        echo "  - Check training log: tail -100 /tmp/periodnet_simple_hybrid.log"
        echo "  - List checkpoints: ls -lh checkpoints/periodnet_simple_hybrid/"
        exit 1
    fi

    remaining=$((max_wait_seconds - elapsed))
    percent=$((100 * elapsed / max_wait_seconds))
    echo "⏳ [$percent%] Waiting... ($elapsed/$max_wait_seconds seconds) [$remaining seconds remaining]"

    sleep $check_interval
    elapsed=$((elapsed + check_interval))
done

echo ""
echo "✅ CHECKPOINT DETECTED!"
echo ""
echo "Checkpoint details:"
ls -lh "$BEST_CHECKPOINT"
echo ""

# Stage-2: LightGBM Training
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Stage-2: LightGBM LambdaRank Training (with corrected CNN features)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Command:"
echo "  python3 -m lc_pipeline.period.train_period_lgbm \\"
echo "    --split-file manifests/damit_gold_split.json \\"
echo "    --manifest data/universe_damit_50k/manifest.csv \\"
echo "    --cnn-checkpoint checkpoints/periodnet_simple_hybrid/best_model.pt \\"
echo "    --output-model artifacts/lgbm_hybrid_fold0.txt \\"
echo "    --device cuda \\"
echo "    --max-cands 64 --num-leaves 64 --learning-rate 0.05 \\"
echo "    --min-data-in-leaf 10 --num-boost-round 500 --early-stopping 50 \\"
echo "    --relabelling-mode focus5"
echo ""

cd "$WORK_DIR"

python3 -m lc_pipeline.period.train_period_lgbm \
    --split-file manifests/damit_gold_split.json \
    --manifest data/universe_damit_50k/manifest.csv \
    --cnn-checkpoint checkpoints/periodnet_simple_hybrid/best_model.pt \
    --output-model artifacts/lgbm_hybrid_fold0.txt \
    --device cuda \
    --max-cands 64 \
    --num-leaves 64 \
    --learning-rate 0.05 \
    --min-data-in-leaf 10 \
    --num-boost-round 500 \
    --early-stopping 50 \
    --relabelling-mode focus5 \
    2>&1 | tee "$LOG_DIR/lgbm_hybrid_training.log"

if [ ! -f "artifacts/lgbm_hybrid_fold0.txt" ]; then
    echo ""
    echo "❌ Stage-2 FAILED: LightGBM model not created"
    echo "Check log: tail -100 $LOG_DIR/lgbm_hybrid_training.log"
    exit 1
fi

echo ""
echo "✅ Stage-2 COMPLETE: LightGBM model saved"
echo ""

# Stage-3: Soft Voting Evaluation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Stage-3: Soft Voting Ensemble Evaluation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Command:"
echo "  python3 -m lc_pipeline.period.eval_period \\"
echo "    --split-file manifests/damit_gold_split.json \\"
echo "    --manifest data/universe_damit_50k/manifest.csv \\"
echo "    --cnn-checkpoints checkpoints/periodnet_simple_hybrid/best_model.pt \\"
echo "    --lgbm-models artifacts/lgbm_hybrid_fold0.txt \\"
echo "    --split test --device cuda \\"
echo "    --metrics-out results/periodnet_v10_hybrid_metrics.json"
echo ""

python3 -m lc_pipeline.period.eval_period \
    --split-file manifests/damit_gold_split.json \
    --manifest data/universe_damit_50k/manifest.csv \
    --cnn-checkpoints checkpoints/periodnet_simple_hybrid/best_model.pt \
    --lgbm-models artifacts/lgbm_hybrid_fold0.txt \
    --split test \
    --device cuda \
    --metrics-out results/periodnet_v10_hybrid_metrics.json \
    2>&1 | tee "$LOG_DIR/eval_hybrid.log"

if [ ! -f "results/periodnet_v10_hybrid_metrics.json" ]; then
    echo ""
    echo "❌ Stage-3 FAILED: Metrics file not created"
    echo "Check log: tail -100 $LOG_DIR/eval_hybrid.log"
    exit 1
fi

echo ""
echo "✅ Stage-3 COMPLETE: Evaluation metrics saved"
echo ""

# Display final results
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PIPELINE COMPLETE: All Stages Successful!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Final Results:"
echo ""

if [ -f "results/periodnet_v10_hybrid_metrics.json" ]; then
    echo "✓ Metrics:"
    cat "results/periodnet_v10_hybrid_metrics.json" | python3 -m json.tool
    echo ""
fi

echo "Artifacts created:"
echo "  ✓ CNN Checkpoint: checkpoints/periodnet_simple_hybrid/best_model.pt"
echo "  ✓ LGBM Model: artifacts/lgbm_hybrid_fold0.txt"
echo "  ✓ LGBM Metadata: artifacts/lgbm_hybrid_fold0.json"
echo "  ✓ Metrics: results/periodnet_v10_hybrid_metrics.json"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
