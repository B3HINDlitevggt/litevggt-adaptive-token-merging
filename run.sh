#!/bin/bash

# =====================================================================
# 실험 환경 기본 설정
# =====================================================================
GT_PATH="./data_scannet_01/scene0001_00_vh_clean.ply"
IMG_DIR="./data_scannet_01/images"
GPU_ID=3

# 순회할 조건 리스트
FRAMES_LIST=(950 975 1000)
CAL_LAYER_MODES=(1 4)
QT_THRESHES=(0.85 0.90)

echo "=========================================================="
echo " LiteVGGT Quadtree-Bipartite 전체 자동화 실험"
echo "  Frames × cal_layer × qt_thresh = $((${#FRAMES_LIST[@]} * ${#CAL_LAYER_MODES[@]} * ${#QT_THRESHES[@]})) 조합"
echo "=========================================================="
echo "대상 데이터셋  : $IMG_DIR"
echo "GPU 디바이스   : $GPU_ID"
echo "프레임 수      : ${FRAMES_LIST[@]}"
echo "cal_layer_mode : ${CAL_LAYER_MODES[@]}"
echo "qt_thresh      : ${QT_THRESHES[@]}"

# =====================================================================
# 1단계 루프: 프레임 수 변경
# =====================================================================
for FRAME in "${FRAMES_LIST[@]}"; do
    echo ""
    echo "##########################################################"
    echo "  [FRAME COUNT: ${FRAME}장 세트 루프 시작]"
    echo "##########################################################"

    # 해당 프레임 크기에 종속될 고유 baseline 폴더 정의
    BASE_LINE_DIR="./output/baseline_scannet_12_f${FRAME}"

    # =================================================================
    # 2단계 루프: cal_layer_mode 변경 (1번 / 4번 재계산)
    # =================================================================
    for CAL in "${CAL_LAYER_MODES[@]}"; do

        # =============================================================
        # 3단계 루프: qt_spatial_thresh 변경 (0.85 / 0.90)
        # =============================================================
        for QT_TH in "${QT_THRESHES[@]}"; do

            # qt_thresh를 파일명용으로 정리 (0.85 → 085)
            QT_TAG=$(echo "$QT_TH" | sed 's/\.//g; s/^0//')

            OUT_DIR="./output/qt_bipartite_t${QT_TAG}_cal${CAL}_scannet_12_f${FRAME}"

            echo ""
            echo "----------------------------------------------------------"
            echo " >> RUNNING: [프레임: ${FRAME}] | [cal_mode: ${CAL}] | [qt_thresh: ${QT_TH}]"
            echo "    output_dir = $OUT_DIR"
            echo "----------------------------------------------------------"

            CUDA_VISIBLE_DEVICES=$GPU_ID python3 run_experiment.py \
                --img_dir "$IMG_DIR" \
                --output_dir "$OUT_DIR" \
                --gt_path "$GT_PATH" \
                --max_frames $FRAME \
                --mode quadtree_bipartite \
                --qt_spatial_thresh $QT_TH \
                --qt_root_block_size 8 \
                --qt_min_block_size 2 \
                --cal_layer_mode $CAL \
                --baseline_dir "$BASE_LINE_DIR"

        done
    done
done

echo ""
echo "=========================================================="
echo " 총 $((${#FRAMES_LIST[@]} * ${#CAL_LAYER_MODES[@]} * ${#QT_THRESHES[@]}))개 매트릭스 조합 실험 종료"
echo "=========================================================="