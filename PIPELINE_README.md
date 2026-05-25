# Unified LiteVGGT Pipeline

This branch integrates the four project branches into one LiteVGGT-based
pipeline while keeping each method selectable by explicit options.

## Integrated Branches

Base branch:

- `main`: STTM / TTT3R-oriented LiteVGGT code path.

Integrated feature branches:

- `jisu`: depth-aware GA token protection using external Depth Anything V2
  boundary priors.
- `jingai`: Sobel and CLS based frame/aggregator variants selected by
  `VGGT_AGGREGATOR`.
- `seojin`: adaptive cache scheduler for reusing merge indices across global
  attention layers.

Important integration choice:

- `seojin` was not merged as a full tree because it is an unrelated initial
  commit and would delete many existing files from `main/jisu`. Only the
  adaptive cache component needed for the unified pipeline was imported:
  `vggt/merging/adaptive_cache.py`.

## Pipeline Structure

The default pipeline remains the LiteVGGT model:

```text
images
  -> VGGT
  -> Aggregator
  -> alternating frame/global attention blocks
  -> optional token merging in global attention
  -> camera/depth/point heads
```

This branch extends the global attention token merging stage with four
independent option groups.

### 1. Depth-Aware GA Token Protection

Files:

- `eval/eval_DTU.py`
- `eval/data.py`
- `merging/merge.py`
- `vggt/layers/block.py`
- `vggt/models/aggregator.py`
- `vggt/models/vggt.py`
- `scripts/cache_da2_depth_boundary.py`
- `scripts/run_eval_dtu.sh`
- `scripts/run_ga_seed_sweep.sh`
- `scripts/save_dtu_qualitative.py`

Method:

```text
base_score = edge_weight * RGB_edge + variance_weight * token_variance
depth_score = Depth Anything V2 boundary map
final_score = base_score + depth_boundary_weight * depth_score
```

Depth boundary maps are cached offline and loaded during evaluation via
`GA_DEPTH_DIR`.

Main recommended settings:

```bash
# best mean candidate
export GA_EDGE_WEIGHT=0.644
export GA_VARIANCE_WEIGHT=0.276
export GA_DEPTH_BOUNDARY_WEIGHT=0.080
export GA_PROTECT_BASE_RATIO=0.10
export GA_DEPTH_PROTECT_RATIO=0.0
```

```bash
# best stability candidate
export GA_EDGE_WEIGHT=0.630
export GA_VARIANCE_WEIGHT=0.270
export GA_DEPTH_BOUNDARY_WEIGHT=0.100
export GA_PROTECT_BASE_RATIO=0.10
export GA_DEPTH_PROTECT_RATIO=0.0
```

Budget split is also implemented:

```bash
# baseline top 9% + depth boundary top 1%
export GA_EDGE_WEIGHT=0.7
export GA_VARIANCE_WEIGHT=0.3
export GA_DEPTH_BOUNDARY_WEIGHT=0.0
export GA_PROTECT_BASE_RATIO=0.10
export GA_DEPTH_PROTECT_RATIO=0.01
```

In current experiments, budget split is treated as future work unless new
results support it.

### 2. Sobel / CLS Aggregator Variants

Files:

- `vggt/models/aggregator_sobel.py`
- `vggt/models/aggregator_cls.py`
- `run_experiment.sh`

Selection is controlled by `VGGT_AGGREGATOR`.

```bash
# default integrated LiteVGGT aggregator
unset VGGT_AGGREGATOR

# Sobel-based aggregator from jingai branch
export VGGT_AGGREGATOR=sobel

# CLS-token aggregator from jingai branch
export VGGT_AGGREGATOR=cls
```

The default is intentionally `default`, not `sobel`, so existing LiteVGGT /
depth-aware GA experiments remain reproducible.

### 3. Adaptive Cache

Files:

- `vggt/merging/adaptive_cache.py`
- `vggt/models/aggregator.py`
- `vggt/models/vggt.py`
- `vggt/layers/block.py`
- `merging/merge.py`

Method:

```text
layer 0 or cold start: recompute merge indices
later layers:
  measure drift of current destination-token features
  if drift > tau or cache_age >= k_max:
      recompute merge indices
  else:
      reuse cached merge indices
```

The adaptive cache is off by default. Enable it from Python:

```python
predictions = model(images, use_adaptive_cache=True)
```

Current default scheduler parameters:

```text
tau_base = 0.25
k_max = 6
use_layer_wise_tau = False
```

### 4. Quadtree-Bipartite Token Merging

Files:

- `merging/sttm_bipartite_merge.py`
- `vggt/layers/block.py`
- `vggt/models/aggregator.py`
- `run_experiment.py`
- `run.sh`

Method:

```text
patch features
  -> recursively split spatial regions with a quadtree rule
  -> use region-level similarity to choose variable-size merge regions
  -> run bipartite token merging with GA protection inside the selected layout
```

This mode is controlled on the model side by:

```python
model.aggregator.use_quadtree_bipartite = True
model.aggregator.qt_spatial_thresh = 0.85
model.aggregator.qt_root_block_size = 8
model.aggregator.qt_min_block_size = 2
```

The updated `run_experiment.py` exposes this as:

```bash
python3 run_experiment.py \
  --img_dir /path/to/images \
  --output_dir ./output/qt_bipartite \
  --gt_path /path/to/gt.ply \
  --mode quadtree_bipartite \
  --qt_spatial_thresh 0.85 \
  --qt_root_block_size 8 \
  --qt_min_block_size 2 \
  --cal_layer_mode 4
```

`--cal_layer_mode` controls how often merge indices are recomputed:

```text
4 -> [0, 6, 15, 20]
3 -> [0, 8, 16]
2 -> [0, 12]
1 -> [0]
```

`run.sh` is a convenience sweep for quadtree-bipartite experiments over:

```text
frame count
cal_layer_mode
qt_spatial_thresh
```

Edit the paths at the top of `run.sh` before running it:

```bash
GT_PATH="./data_scannet_01/scene0001_00_vh_clean.ply"
IMG_DIR="./data_scannet_01/images"
GPU_ID=3
```

## Default DTU Evaluation

Use this for baseline or depth-aware GA experiments.

```bash
cd /path/to/LiteVGGT-repo
source .venv/bin/activate
export PYTHONPATH=$PWD:${PYTHONPATH:-}

export MODEL_PATH=$PWD/checkpoints/te_dict.pt
export DTU_DIR=/data/eval_data/dtu

bash scripts/run_eval_dtu.sh
```

## Final Integrated Pipeline

Use this when evaluating the full project pipeline against LiteVGGT:

```text
Sobel frame selection
  -> default LiteVGGT aggregator
  -> Depth Anything V2 boundary GA score, b008_r010
  -> Quadtree-Bipartite token merging
  -> optional adaptive merge-index cache
  -> camera/depth/point outputs
```

This is intentionally different from `VGGT_AGGREGATOR=sobel`. The environment
aggregator switch replaces the whole aggregator, while the final pipeline keeps
the default aggregator so that depth-aware GA, quadtree-bipartite merging, and
adaptive cache can run together. Sobel is used as an input frame-selection step.

LiteVGGT comparison run:

```bash
python3 run_experiment.py \
  --ckpt_path checkpoints/te_dict.pt \
  --img_dir /data/drive_files/scannetpp1 \
  --output_dir outputs/final_pipeline_scannetpp1/litevggt \
  --mode baseline \
  --max_frames 48 \
  --cal_layer_mode 4
```

Full integrated run:

```bash
python3 run_experiment.py \
  --ckpt_path checkpoints/te_dict.pt \
  --img_dir /data/drive_files/scannetpp1 \
  --output_dir outputs/final_pipeline_scannetpp1/integrated \
  --mode quadtree_bipartite \
  --frame_selection sobel \
  --max_frames 24 \
  --ga_depth_dir /data/eval_data/scannetpp_da2s_depth_boundary/scannetpp1 \
  --ga_edge_weight 0.644 \
  --ga_variance_weight 0.276 \
  --ga_depth_boundary_weight 0.080 \
  --ga_depth_map_is_boundary \
  --ga_protect_base_ratio 0.10 \
  --ga_depth_protect_ratio 0.0 \
  --use_adaptive_cache \
  --qt_spatial_thresh 0.8 \
  --qt_root_block_size 8 \
  --qt_min_block_size 2 \
  --cal_layer_mode 4
```

Each run writes:

```text
experiment_log.txt
frame_log.csv
recon.ply
cd_result.txt, when --gt_path is provided
```

For large ScanNet/ScanNet++ runs, use `--skip_ply` to collect runtime/token
metrics without writing a large point cloud, or cap the visualization size with
`--max_ply_points 2000000`.

## Depth Boundary Cache

Depth Anything V2-Small checkpoint:

```bash
mkdir -p /home/ec2-user/workspace/Depth-Anything-V2/checkpoints
wget -O /home/ec2-user/workspace/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

Cache DTU depth boundary maps:

```bash
python3 scripts/cache_da2_depth_boundary.py \
  --dtu_dir /data/eval_data/dtu \
  --output_dir /data/eval_data/da2s_depth_boundary \
  --checkpoint /home/ec2-user/workspace/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  --repo_dir /home/ec2-user/workspace/Depth-Anything-V2 \
  --encoder vits \
  --scenes scan1,scan6
```

Then set:

```bash
export GA_DEPTH_DIR=/data/eval_data/da2s_depth_boundary
export GA_DEPTH_MAP_IS_BOUNDARY=1
```

## Recommended Experiments

### Baseline

```bash
export EXP_NAME=baseline
export GA_EDGE_WEIGHT=0.7
export GA_VARIANCE_WEIGHT=0.3
export GA_DEPTH_BOUNDARY_WEIGHT=0.0
export GA_PROTECT_BASE_RATIO=0.10
export GA_DEPTH_PROTECT_RATIO=0.0

bash scripts/run_ga_seed_sweep.sh
```

### Depth Boundary Prior: Best Mean

```bash
export EXP_NAME=da2s_depth_b008_r010
export GA_DEPTH_DIR=/data/eval_data/da2s_depth_boundary
export GA_DEPTH_MAP_IS_BOUNDARY=1

export GA_EDGE_WEIGHT=0.644
export GA_VARIANCE_WEIGHT=0.276
export GA_DEPTH_BOUNDARY_WEIGHT=0.080
export GA_PROTECT_BASE_RATIO=0.10
export GA_DEPTH_PROTECT_RATIO=0.0

bash scripts/run_ga_seed_sweep.sh
```

### Depth Boundary Prior: Best Stability

```bash
export EXP_NAME=da2s_depth_b010_r010
export GA_DEPTH_DIR=/data/eval_data/da2s_depth_boundary
export GA_DEPTH_MAP_IS_BOUNDARY=1

export GA_EDGE_WEIGHT=0.630
export GA_VARIANCE_WEIGHT=0.270
export GA_DEPTH_BOUNDARY_WEIGHT=0.100
export GA_PROTECT_BASE_RATIO=0.10
export GA_DEPTH_PROTECT_RATIO=0.0

bash scripts/run_ga_seed_sweep.sh
```

### Budget Split Sanity Check

```bash
export EXP_NAME=da2s_depth_split_9_1_seed0
export SEEDS="0"
export GA_DEPTH_DIR=/data/eval_data/da2s_depth_boundary
export GA_DEPTH_MAP_IS_BOUNDARY=1

export GA_EDGE_WEIGHT=0.7
export GA_VARIANCE_WEIGHT=0.3
export GA_DEPTH_BOUNDARY_WEIGHT=0.0
export GA_PROTECT_BASE_RATIO=0.10
export GA_DEPTH_PROTECT_RATIO=0.01

bash scripts/run_ga_seed_sweep.sh
```

## Qualitative Visualization

DTU qualitative panels:

```bash
python3 scripts/save_dtu_qualitative.py \
  --model_path "$MODEL_PATH" \
  --dtu_dir "$DTU_DIR" \
  --ga_depth_dir "$GA_DEPTH_DIR" \
  --ga_depth_map_is_boundary \
  --ga_edge_weight 0.644 \
  --ga_variance_weight 0.276 \
  --ga_depth_boundary_weight 0.080 \
  --ga_protect_base_ratio 0.10 \
  --scene scan1 \
  --frames 0,8,16,24,32,40 \
  --output_dir outputs/qual_da2s_depth_b008_r010
```

Panels include:

```text
input image
predicted depth
depth confidence
baseline GA score
depth boundary prior
final GA score
```

## Sobel / CLS Aggregator Experiments

Use `run_experiment.sh` from the jingai branch integration.

```bash
# Sobel aggregator
./run_experiment.sh \
  --aggregator sobel \
  --mode baseline \
  --img_dir /path/to/images \
  --ckpt_path /path/to/te_dict.pt \
  --output_base ./output
```

```bash
# CLS aggregator
./run_experiment.sh \
  --aggregator cls \
  --mode baseline \
  --img_dir /path/to/images \
  --ckpt_path /path/to/te_dict.pt \
  --output_base ./output
```

Supported modes in `run_experiment.sh`:

```text
baseline
dynamic_protect
dynamic_grid
dynamic_all
sttm
```

## Quadtree-Bipartite Experiments

The latest `main` branch adds a quadtree-bipartite merge mode to
`run_experiment.py`.

Single run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 run_experiment.py \
  --img_dir /path/to/images \
  --output_dir ./output/qt_bipartite_t085_cal4 \
  --gt_path /path/to/gt.ply \
  --mode quadtree_bipartite \
  --qt_spatial_thresh 0.85 \
  --qt_root_block_size 8 \
  --qt_min_block_size 2 \
  --cal_layer_mode 4 \
  --baseline_dir ./output/baseline
```

Automatic sweep:

```bash
bash run.sh
```

Before using `run.sh`, edit:

```text
GT_PATH
IMG_DIR
GPU_ID
FRAMES_LIST
CAL_LAYER_MODES
QT_THRESHES
```

## Option Summary

Environment variables:

```text
VGGT_AGGREGATOR
  unset/default : integrated default LiteVGGT aggregator
  sobel         : jingai Sobel aggregator
  cls           : jingai CLS aggregator

GA_DEPTH_DIR
  root directory containing scene/image-stem depth boundary `.npy` files

GA_DEPTH_MAP_IS_BOUNDARY
  set to 1 when `GA_DEPTH_DIR` already stores boundary maps

GA_EDGE_WEIGHT
GA_VARIANCE_WEIGHT
GA_DEPTH_BOUNDARY_WEIGHT
  additive GA score weights

GA_PROTECT_BASE_RATIO
  total protected token ratio

GA_DEPTH_PROTECT_RATIO
  separate depth top-k budget for split selection.
  keep at 0.0 for additive depth prior experiments.
```

Python-only option:

```python
model(images, use_adaptive_cache=True)
```

`run_experiment.py` options:

```text
--mode
  baseline
  dynamic_protect
  dynamic_grid
  dynamic_all
  sttm
  quadtree_bipartite

--cal_layer_mode
  4, 3, 2, or 1

--qt_spatial_thresh
--qt_root_block_size
--qt_min_block_size
  quadtree-bipartite controls
```

## Reproducibility Notes

- For the main depth-aware result, keep `GA_DEPTH_PROTECT_RATIO=0.0`.
- `GA_DEPTH_PROTECT_RATIO>0` switches to budget-split selection.
- `VGGT_AGGREGATOR=sobel|cls` uses the alternative jingai aggregator and does
  not use the default depth-aware GA path.
- Adaptive cache is off by default and should be evaluated separately because
  it changes merge-index reuse across layers.
- Quadtree-bipartite is also off by default. It is enabled by
  `--mode quadtree_bipartite` in `run_experiment.py`, or by setting
  `model.aggregator.use_quadtree_bipartite=True`.
- Do not commit generated files such as `logs/`, `outputs/`, `qual_*`,
  `__pycache__/`, `.DS_Store`, checkpoints, or downloaded datasets.
