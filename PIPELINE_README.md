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

This branch extends the global attention token merging stage with three
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

## Reproducibility Notes

- For the main depth-aware result, keep `GA_DEPTH_PROTECT_RATIO=0.0`.
- `GA_DEPTH_PROTECT_RATIO>0` switches to budget-split selection.
- `VGGT_AGGREGATOR=sobel|cls` uses the alternative jingai aggregator and does
  not use the default depth-aware GA path.
- Adaptive cache is off by default and should be evaluated separately because
  it changes merge-index reuse across layers.
- Do not commit generated files such as `logs/`, `outputs/`, `qual_*`,
  `__pycache__/`, `.DS_Store`, checkpoints, or downloaded datasets.
