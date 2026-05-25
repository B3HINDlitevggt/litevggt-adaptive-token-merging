import os
import torch
import numpy as np
import gzip
import json
import random
import logging
import warnings
import cv2
from vggt.models.vggt import VGGT
from vggt.utils.rotation import mat_to_quat
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3
# from ba import run_vggt_with_ba
import argparse

# Suppress DINO v2 logs
logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="dinov2")

# Set computation precision
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False


def convert_pt3d_RT_to_opencv(Rot, Trans):
    """
    Convert Point3D extrinsic matrices to OpenCV convention.

    Args:
        Rot: 3D rotation matrix in Point3D format
        Trans: 3D translation vector in Point3D format

    Returns:
        extri_opencv: 3x4 extrinsic matrix in OpenCV format
    """
    rot_pt3d = np.array(Rot)
    trans_pt3d = np.array(Trans)

    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d = rot_pt3d.transpose(1, 0)
    extri_opencv = np.hstack((rot_pt3d, trans_pt3d[:, None]))
    return extri_opencv


def build_pair_index(N, B=1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using NumPy.

    Args:
        r_error: numpy array representing R error values (Degree)
        t_error: numpy array representing T error values (Degree)
        max_threshold: Maximum threshold value for binning the histogram

    Returns:
        AUC value and the normalized histogram
    """
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.
    This function assumes the input poses are world-to-camera (w2c) transformations.

    Args:
        pred_se3: Predicted SE(3) transformations (w2c), shape (N, 4, 4)
        gt_se3: Ground truth SE(3) transformations (w2c), shape (N, 4, 4)
        num_frames: Number of frames (N)

    Returns:
        Rotation and translation angle errors in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    relative_pose_gt = gt_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(gt_se3[pair_idx_i2])
    )
    relative_pose_pred = pred_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(pred_se3[pair_idx_i2])
    )

    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg


def setup_args():
    """Set up command-line arguments for the CO3D evaluation script."""
    parser = argparse.ArgumentParser(description='Test VGGT on CO3D dataset')
    parser.add_argument('--dtu_dir', type=str, required=True, help='Path to tnt dir')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the VGGT model checkpoint')
    parser.add_argument('--ga_depth_dir', type=str, default=None, help='Optional root dir of external depth maps for GA metric')
    parser.add_argument('--ga_edge_weight', type=float, default=0.7, help='Weight for image edge score in GA metric')
    parser.add_argument('--ga_variance_weight', type=float, default=0.3, help='Weight for patch variance score in GA metric')
    parser.add_argument('--ga_depth_boundary_weight', type=float, default=0.0, help='Weight for depth boundary score in GA metric')
    parser.add_argument('--ga_depth_map_is_boundary', action='store_true', help='Treat loaded GA depth maps as precomputed boundary maps')
    parser.add_argument('--ga_interaction_weight', type=float, default=0.0, help='Weight for edge-variance interaction score in GA metric')
    parser.add_argument('--ga_interaction_mode', choices=['sqrt', 'product'], default='sqrt', help='Interaction score mode for GA metric')
    parser.add_argument('--ga_laplacian_weight', type=float, default=0.0, help='Weight for image Laplacian proxy score in GA metric')
    parser.add_argument('--ga_adaptive_weights', action='store_true', help='Adapt edge/variance weights from frame-level map means')
    parser.add_argument('--ga_adaptive_protect_ratio', action='store_true', help='Adapt protected GA token ratio from frame complexity')
    parser.add_argument('--ga_protect_base_ratio', type=float, default=0.1, help='Base protected token ratio for GA')
    parser.add_argument('--ga_protect_complexity_lambda', type=float, default=0.0, help='Complexity scale for adaptive protected token ratio')
    parser.add_argument('--ga_protect_min_ratio', type=float, default=0.05, help='Minimum protected token ratio when adaptive protection is enabled')
    parser.add_argument('--ga_protect_max_ratio', type=float, default=0.2, help='Maximum protected token ratio when adaptive protection is enabled')
    parser.add_argument('--ga_protect_nms', action='store_true', help='Select protected GA tokens from local maxima in the info map')
    parser.add_argument('--ga_depth_protect_ratio', type=float, default=0.0, help='Absolute protected token ratio reserved for depth boundary top-k')
    return parser.parse_args()


def find_depth_file(depth_scene_dir, image_name):
    stem = os.path.splitext(image_name)[0]
    candidate_names = [
        f"{stem}.npy",
        f"{stem}.npz",
        f"{stem}.png",
        f"{stem}.jpg",
        f"{stem}.jpeg",
        f"{stem}.tif",
        f"{stem}.tiff",
        f"{stem}_depth.npy",
        f"{stem}_depth.npz",
        f"{stem}_depth.png",
        f"{stem}_depth.jpg",
        f"{stem}_depth.jpeg",
        f"{stem}_depth.tif",
        f"{stem}_depth.tiff",
    ]

    for candidate_name in candidate_names:
        candidate_path = os.path.join(depth_scene_dir, candidate_name)
        if os.path.isfile(candidate_path):
            return candidate_path

    return None


def load_depth_file(depth_path):
    if depth_path.endswith(".npy"):
        depth = np.load(depth_path)
    elif depth_path.endswith(".npz"):
        depth_npz = np.load(depth_path)
        if "depth" in depth_npz:
            depth = depth_npz["depth"]
        else:
            first_key = list(depth_npz.keys())[0]
            depth = depth_npz[first_key]
    else:
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return depth


def build_ga_depth_tensor(ga_depth_dir, scene, image_paths, target_hw):
    if ga_depth_dir is None:
        return None

    depth_scene_dir = os.path.join(ga_depth_dir, scene)
    if not os.path.isdir(depth_scene_dir):
        raise FileNotFoundError(f"GA depth scene directory not found: {depth_scene_dir}")

    target_h, target_w = target_hw
    depth_tensors = []
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        depth_path = find_depth_file(depth_scene_dir, image_name)
        if depth_path is None:
            raise FileNotFoundError(
                f"No GA depth file found for image '{image_name}' under {depth_scene_dir}"
            )

        depth = load_depth_file(depth_path)
        if depth.shape != (target_h, target_w):
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        depth_tensors.append(torch.from_numpy(depth))

    stacked = torch.stack(depth_tensors, dim=0)
    print(
        f"Loaded GA depth maps: scene={scene}, count={len(depth_tensors)}, "
        f"shape={tuple(stacked.shape)}, min={stacked.min().item():.4f}, "
        f"max={stacked.max().item():.4f}, mean={stacked.mean().item():.4f}"
    )
    return stacked


def load_model(device, model_path):
    """
    Load the VGGT model.

    Args:
        device: Device to load the model on
        model_path: Path to the model checkpoint

    Returns:
        Loaded VGGT model
    """
    print("Initializing and loading VGGT model...")
    model = VGGT()
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    print(f"USING {model_path}")
    model = VGGT().to(device)
    ckpt_path = model_path
    # ckpt_path = "te_dict.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint: 
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint 
    model.load_state_dict(state_dict, strict=False)  
    model = model.to(torch.bfloat16)

    print("Model loaded")
    
    return model


def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def process_sequence(model, seq_name, seq_data, category, co3d_dir, min_num_images, num_frames, use_ba, device, dtype):
    """
    Process a single sequence and compute pose errors.

    Args:
        model: VGGT model
        seq_name: Sequence name
        seq_data: Sequence data
        category: Category name
        co3d_dir: CO3D dataset directory
        min_num_images: Minimum number of images required
        num_frames: Number of frames to sample
        use_ba: Whether to use bundle adjustment
        device: Device to run on
        dtype: Data type for model inference

    Returns:
        rError: Rotation errors
        tError: Translation errors
    """
    if len(seq_data) < min_num_images:
        return None, None

    metadata = []
    for data in seq_data:
        # Make sure translations are not ridiculous
        if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
            return None, None
        extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])
        metadata.append({
            "filepath": data["filepath"],
            "extri": extri_opencv,
        })

    # Random sample num_frames images
    ids = np.random.choice(len(metadata), num_frames, replace=False)
    print("Image ids", ids)

    image_names = [os.path.join(co3d_dir, metadata[i]["filepath"]) for i in ids]
    gt_extri = [np.array(metadata[i]["extri"]) for i in ids]
    gt_extri = np.stack(gt_extri, axis=0)

    images = load_and_preprocess_images(image_names).to(device)

    patch_width = images.shape[-1] // 14
    patch_height = images.shape[-2] // 14
    model.update_patch_dimensions(patch_width, patch_height)

    with torch.no_grad():
        predictions = model(images)

    with torch.amp.autocast("cuda",dtype=torch.float64):
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        pred_extrinsic = extrinsic[0]

    with torch.amp.autocast("cuda",dtype=torch.float64):
        gt_extrinsic = torch.from_numpy(gt_extri).to(device)
        add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)

        pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
        gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)

        Racc_5 = (rel_rangle_deg < 5).float().mean().item()
        Tacc_5 = (rel_tangle_deg < 5).float().mean().item()

        print(f"{category} sequence {seq_name} R_ACC@5: {Racc_5:.4f}")
        print(f"{category} sequence {seq_name} T_ACC@5: {Tacc_5:.4f}")

        return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy()


def main():
    """Main function to evaluate VGGT on CO3D dataset."""
    # Parse command-line arguments
    args = setup_args()

    # Setup device and data type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Load model
    model = load_model(device, model_path=args.model_path)
    model.set_ga_metric_weights(
        edge_weight=args.ga_edge_weight,
        variance_weight=args.ga_variance_weight,
        depth_boundary_weight=args.ga_depth_boundary_weight,
        depth_map_is_boundary=args.ga_depth_map_is_boundary,
        interaction_weight=args.ga_interaction_weight,
        interaction_mode=args.ga_interaction_mode,
        laplacian_weight=args.ga_laplacian_weight,
        adaptive_weights=args.ga_adaptive_weights,
        adaptive_protect_ratio=args.ga_adaptive_protect_ratio,
        protect_base_ratio=args.ga_protect_base_ratio,
        protect_complexity_lambda=args.ga_protect_complexity_lambda,
        protect_min_ratio=args.ga_protect_min_ratio,
        protect_max_ratio=args.ga_protect_max_ratio,
        protect_nms=args.ga_protect_nms,
        depth_protect_ratio=args.ga_depth_protect_ratio,
    )
    print(
        "GA config: "
        f"edge={args.ga_edge_weight}, variance={args.ga_variance_weight}, "
        f"depth_boundary={args.ga_depth_boundary_weight}, "
        f"depth_dir={args.ga_depth_dir}, depth_is_boundary={args.ga_depth_map_is_boundary}, "
        f"protect_base={args.ga_protect_base_ratio}, depth_protect={args.ga_depth_protect_ratio}"
    )

    # Set random seeds
    set_random_seeds(args.seed)

    per_category_results = {}



    from data import SevenScenes,NRGBD,TnTDataset,DTUDataset


    dataset = DTUDataset(root_dir=args.dtu_dir)

    for sample in dataset:
        category = sample['scene']
        print(f"eval {category}!")
        # B 3 H W 0-1
        images = sample["imgs"].to(device)
        c2w_gt = sample["poses"].to(device)
        image_paths = sample["image_paths"]
        max_frames = 48
        images = images[:max_frames]
        c2w_gt = c2w_gt[:max_frames]
        image_paths = image_paths[:max_frames]
        N_aligned = images.shape[0]

        print(f"✅ images: {images.shape}, poses: {c2w_gt.shape}")

        # w2c_gt = np.linalg.inv(c2w_gt.cpu().numpy())
        w2c_gt = c2w_gt.cpu().numpy()
        rError = []
        tError = []


        patch_width = images.shape[-1] // 14
        patch_height = images.shape[-2] // 14
        model.update_patch_dimensions(patch_width, patch_height)

        ga_depth = build_ga_depth_tensor(
            args.ga_depth_dir,
            category,
            image_paths,
            target_hw=images.shape[-2:],
        )
        if ga_depth is not None:
            ga_depth = ga_depth.unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(images, ga_depth=ga_depth)

            with torch.amp.autocast("cuda",dtype=torch.float64):
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
                pred_extrinsic = extrinsic[0]

            with torch.amp.autocast("cuda",dtype=torch.float64):
                gt_se3 = torch.from_numpy(w2c_gt).to(device)
                add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)

                pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
                # gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

                print(gt_se3.shape,pred_se3.shape)

                rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, N_aligned)

                Racc_5 = (rel_rangle_deg < 5).float().mean().item()
                Tacc_5 = (rel_tangle_deg < 5).float().mean().item()

                print(f"{category} sequence  R_ACC@5: {Racc_5:.4f}")
                print(f"{category} sequence  T_ACC@5: {Tacc_5:.4f}")

                rError = rel_rangle_deg.cpu().detach().numpy()
                tError = rel_tangle_deg.cpu().detach().numpy()


        rError = np.array(rError)
        tError = np.array(tError)

        Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
        Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
        Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
        Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)

        per_category_results[category] = {
            "rError": rError,
            "tError": tError,
            "Auc_30": Auc_30,
            "Auc_15": Auc_15,
            "Auc_5": Auc_5,
            "Auc_3": Auc_3
        }

        print("="*80)
        # Print results with colors
        GREEN = "\033[92m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        print(f"{BOLD}{BLUE}AUC of {category} test set:{RESET} {GREEN}{Auc_30:.4f} (AUC@30), {Auc_15:.4f} (AUC@15), {Auc_5:.4f} (AUC@5), {Auc_3:.4f} (AUC@3){RESET}")
        mean_AUC_30_by_now = np.mean([per_category_results[category]["Auc_30"] for category in per_category_results])
        mean_AUC_15_by_now = np.mean([per_category_results[category]["Auc_15"] for category in per_category_results])
        mean_AUC_5_by_now = np.mean([per_category_results[category]["Auc_5"] for category in per_category_results])
        mean_AUC_3_by_now = np.mean([per_category_results[category]["Auc_3"] for category in per_category_results])
        print(f"{BOLD}{BLUE}Mean AUC of categories by now:{RESET} {RED}{mean_AUC_30_by_now:.4f} (AUC@30), {mean_AUC_15_by_now:.4f} (AUC@15), {mean_AUC_5_by_now:.4f} (AUC@5), {mean_AUC_3_by_now:.4f} (AUC@3){RESET}")
        print("="*80)

    # Print summary results
    print("\nSummary of AUC results:")
    print("-"*50)
    for category in sorted(per_category_results.keys()):
        print(f"{category:<15}: {per_category_results[category]['Auc_30']:.4f} (AUC@30), {per_category_results[category]['Auc_15']:.4f} (AUC@15), {per_category_results[category]['Auc_5']:.4f} (AUC@5), {per_category_results[category]['Auc_3']:.4f} (AUC@3)")

    if per_category_results:
        mean_AUC_30 = np.mean([per_category_results[category]["Auc_30"] for category in per_category_results])
        mean_AUC_15 = np.mean([per_category_results[category]["Auc_15"] for category in per_category_results])
        mean_AUC_5 = np.mean([per_category_results[category]["Auc_5"] for category in per_category_results])
        mean_AUC_3 = np.mean([per_category_results[category]["Auc_3"] for category in per_category_results])
        print("-"*50)
        print(f"Mean AUC: {mean_AUC_30:.4f} (AUC@30), {mean_AUC_15:.4f} (AUC@15), {mean_AUC_5:.4f} (AUC@5), {mean_AUC_3:.4f} (AUC@3)")
    print(args.model_path)

if __name__ == "__main__":
    main()
