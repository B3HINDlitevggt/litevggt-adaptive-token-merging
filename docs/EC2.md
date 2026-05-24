# EC2 Deployment

This repo can run on AWS EC2 with a GPU instance. The lowest-friction path is:

- Use an Ubuntu 22.04 GPU AMI that already includes NVIDIA driver and CUDA.
- Prefer an x86 GPU instance, not ARM, because the repo dependencies are set up for the standard PyTorch CUDA stack.
- Start with `g5.2xlarge` for single-GPU evaluation. Move to `g5.4xlarge` or larger if you hit memory or throughput limits.

## 1. Launch the instance

Recommended baseline:

- Instance family: `g5`
- AMI: AWS Deep Learning OSS Nvidia Driver AMI for Ubuntu 22.04
- Root volume: at least `200 GB` gp3 if you plan to store datasets locally
- Security group: allow `22/tcp` from your IP only

## 2. Upload the repo

From your local machine:

```bash
scp -i /path/to/key.pem -r LiteVGGT-repo ubuntu@YOUR_EC2_PUBLIC_DNS:~/
```

If the repo is already on GitHub:

```bash
ssh -i /path/to/key.pem ubuntu@YOUR_EC2_PUBLIC_DNS
git clone <your-repo-url>
cd LiteVGGT-repo
```

## 3. Install dependencies

On the EC2 instance:

```bash
cd ~/LiteVGGT-repo
bash scripts/setup_ec2.sh
```

If you want the script to also download the checkpoint:

```bash
cd ~/LiteVGGT-repo
export CKPT_URL="https://huggingface.co/ZhijianShu/LiteVGGT/resolve/main/te_dict.pt"
bash scripts/setup_ec2.sh
```

## 4. Upload datasets and depth maps

Suggested layout:

```text
~/data/dtu/<scene>/...
~/data/dtu_depth/<scene>/<image_stem>.npy
```

You can upload data with `scp` or `rsync`. Example:

```bash
rsync -avz -e "ssh -i /path/to/key.pem" /local/path/eval_data/dtu/ ubuntu@YOUR_EC2_PUBLIC_DNS:~/data/dtu/
rsync -avz -e "ssh -i /path/to/key.pem" /local/path/dtu_depth/ ubuntu@YOUR_EC2_PUBLIC_DNS:~/data/dtu_depth/
```

## 5. Run DTU evaluation

Baseline:

```bash
cd ~/LiteVGGT-repo
export MODEL_PATH=~/LiteVGGT-repo/checkpoints/te_dict.pt
export DTU_DIR=~/data/dtu
bash scripts/run_eval_dtu.sh
```

Depth-boundary GA experiment:

```bash
cd ~/LiteVGGT-repo
export MODEL_PATH=~/LiteVGGT-repo/checkpoints/te_dict.pt
export DTU_DIR=~/data/dtu
export GA_DEPTH_DIR=~/data/dtu_depth
export GA_EDGE_WEIGHT=0.5
export GA_VARIANCE_WEIGHT=0.2
export GA_DEPTH_BOUNDARY_WEIGHT=0.3
export GA_INTERACTION_WEIGHT=0.0
bash scripts/run_eval_dtu.sh
```

Edge + variance interaction GA experiment:

```bash
cd ~/LiteVGGT-repo
export MODEL_PATH=~/LiteVGGT-repo/checkpoints/te_dict.pt
export DTU_DIR=~/data/dtu
export GA_EDGE_WEIGHT=0.5
export GA_VARIANCE_WEIGHT=0.3
export GA_INTERACTION_WEIGHT=0.2
bash scripts/run_eval_dtu.sh
```

## 6. Keep jobs alive after SSH disconnect

Use `tmux`:

```bash
sudo apt-get install -y tmux
tmux new -s litevggt
cd ~/LiteVGGT-repo
bash scripts/run_eval_dtu.sh
```

Detach with `Ctrl-b d`, reattach with:

```bash
tmux attach -t litevggt
```

## Notes

- `Transformer Engine` is the most likely install failure point. If it fails, check `nvidia-smi`, CUDA visibility, and compiler toolchain first.
- The evaluation script expects the depth directory to be organized by scene name.
- Depth file names are matched using the RGB image stem, for example `000000.png -> 000000.npy`.
