#!/usr/bin/env python3
"""
Robust reconstruction script (A + B fixes).

Improvements B applied:
 - dtype consistency
 - mutual (cross) matching
 - subpixel refinement of keypoints for triangulation
 - undistortPoints for geometric ops
 - safe rounding & color clipping
 - descriptor dtype enforcement for FLANN
 - slightly stricter matching defaults (ratio 0.75)
"""

import glob
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False
    print("Warning: open3d not available - interactive view disabled.")


def filter_points_by_triangulation(pts3d, R1, t1, R2, t2, K, pts1, pts2,
                                   max_reproj_error=2.0, min_angle_deg=0.3, max_distance=200.0):
    """Filter 3D points with threshold checks (safe divisions and guards)."""
    # Ensure numpy dtypes
    pts3d = np.asarray(pts3d, dtype=np.float64)
    R1 = np.asarray(R1, dtype=np.float64)
    t1 = np.asarray(t1, dtype=np.float64)
    R2 = np.asarray(R2, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))

    n = len(pts3d)
    valid_mask = np.ones(n, dtype=bool)
    reasons = {'depth': 0, 'reproj': 0, 'angle': 0, 'distance': 0}

    pts3d_h = np.hstack((pts3d, np.ones((n, 1), dtype=np.float64)))

    # Positive depth in both cameras
    pts_cam1 = (R1 @ pts3d.T + t1).T
    pts_cam2 = (R2 @ pts3d.T + t2).T
    depth1_check = pts_cam1[:, 2] > 0
    depth2_check = pts_cam2[:, 2] > 0
    bad_depth = ~(depth1_check & depth2_check)
    reasons['depth'] += np.sum(bad_depth)
    valid_mask &= (depth1_check & depth2_check)

    # Camera centers
    cam1_center = (-R1.T @ t1).reshape(3)
    cam2_center = (-R2.T @ t2).reshape(3)

    # Distance check
    dist1 = np.linalg.norm(pts3d - cam1_center[None, :], axis=1)
    dist2 = np.linalg.norm(pts3d - cam2_center[None, :], axis=1)
    dist_check = (dist1 < max_distance) & (dist2 < max_distance)
    reasons['distance'] += np.sum(~dist_check)
    valid_mask &= dist_check

    # Reprojection error (safe divide)
    pts_proj1 = (P1 @ pts3d_h.T).T
    z1 = pts_proj1[:, 2:3].copy()
    z1[np.abs(z1) < 1e-8] = 1e-8
    pts_proj1_xy = pts_proj1[:, :2] / z1
    error1 = np.linalg.norm(pts_proj1_xy - pts1, axis=1)

    pts_proj2 = (P2 @ pts3d_h.T).T
    z2 = pts_proj2[:, 2:3].copy()
    z2[np.abs(z2) < 1e-8] = 1e-8
    pts_proj2_xy = pts_proj2[:, :2] / z2
    error2 = np.linalg.norm(pts_proj2_xy - pts2, axis=1)

    reproj_check = (error1 < max_reproj_error) & (error2 < max_reproj_error)
    reasons['reproj'] += np.sum(~reproj_check)
    valid_mask &= reproj_check

    # Triangulation angle check (guard near-zero norms)
    for i in range(n):
        if not valid_mask[i]:
            continue
        ray1 = pts3d[i] - cam1_center
        ray2 = pts3d[i] - cam2_center
        norm1 = np.linalg.norm(ray1)
        norm2 = np.linalg.norm(ray2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            valid_mask[i] = False
            reasons['angle'] += 1
            continue
        ray1n = ray1 / norm1
        ray2n = ray2 / norm2
        cosang = np.clip(np.dot(ray1n, ray2n), -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cosang)))
        if angle_deg < min_angle_deg or angle_deg > 180.0 - min_angle_deg:
            valid_mask[i] = False
            reasons['angle'] += 1

    return valid_mask, reasons


def mutual_matches(desA, desB, flann, ratio_thresh=0.75):
    """
    Return mutual matches between A->B and B->A using FLANN knn and ratio test.
    Returns list of cv2.DMatch where queryIdx refers to A and trainIdx refers to B.
    """
    # ensure float32
    desA_f = desA.astype(np.float32)
    desB_f = desB.astype(np.float32)

    knn_ab = flann.knnMatch(desA_f, desB_f, k=2)
    good_ab = {}
    for m_n in knn_ab:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            good_ab[m.queryIdx] = m.trainIdx

    knn_ba = flann.knnMatch(desB_f, desA_f, k=2)
    good_ba = {}
    for m_n in knn_ba:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            good_ba[m.queryIdx] = m.trainIdx

    # mutual intersection
    mutual = []
    for a_idx, b_idx in good_ab.items():
        if b_idx in good_ba and good_ba[b_idx] == a_idx:
            # create DMatch with indices aligned to A->B
            m = cv2.DMatch(_queryIdx=a_idx, _trainIdx=b_idx, _imgIdx=0, _distance=0.0)
            mutual.append(m)
    return mutual


def refine_keypoints_subpix(gray_img, pts, win_size=(5,5), zero_zone=(-1,-1), criteria=None):
    """
    Refine 2D points to subpixel accuracy for better triangulation.
    pts: Nx2 float32 array of pixel locations
    returns Nx2 refined points (float32)
    """
    if criteria is None:
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 1e-4)
    if len(pts) == 0:
        return pts
    # cornerSubPix expects points as Nx1x2 float32
    pts_cv = pts.reshape(-1, 1, 2).astype(np.float32)
    cv2.cornerSubPix(gray_img, pts_cv, win_size, zero_zone, criteria)
    return pts_cv.reshape(-1, 2)


def reconstruct_sequence(image_files, K,
                         dist_coeffs=None,
                         max_reproj_error=0.8, min_angle_deg=1.0, max_distance=None,
                         min_matches=12, flann_checks=64, ratio_thresh=0.75):
    """
    Reconstruct pairwise along the image sequence.
    Returns points3D (Nx3), colors (Nx3), camera_poses list [(R,t), ...]
    """
    if len(image_files) < 2:
        raise ValueError("Need at least two images to reconstruct.")

    # Create SIFT + FLANN
    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=flann_checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Read first image
    img_prev = cv2.imread(image_files[0])
    if img_prev is None:
        raise FileNotFoundError(f"Can't read image {image_files[0]}")
    gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    kp_prev, des_prev = sift.detectAndCompute(gray_prev, None)

    # initial pose (world = first camera)
    R_prev = np.eye(3, dtype=np.float64)
    t_prev = np.zeros((3, 1), dtype=np.float64)
    camera_poses = [(R_prev.copy(), t_prev.copy())]

    points3D_all = []
    colors_all = []

    total_raw = 0
    successful_pairs = 0

    for i in range(1, len(image_files)):
        print(f"\n--- Pair {i-1} -> {i} ---")
        img_curr = cv2.imread(image_files[i])
        if img_curr is None:
            print(f"Warning: Can't read {image_files[i]}, skipping.")
            continue
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
        kp_curr, des_curr = sift.detectAndCompute(gray_curr, None)

        # Safety: descriptors available
        if des_prev is None or des_curr is None or len(kp_prev) == 0 or len(kp_curr) == 0:
            print("No descriptors in one image; skipping.")
            kp_prev, des_prev = kp_curr, des_curr
            img_prev, gray_prev = img_curr, gray_curr
            continue

        # Mutual matches (cross-check) to reduce false matches
        matches = mutual_matches(des_prev, des_curr, flann, ratio_thresh=ratio_thresh)
        print(f"Mutual matches: {len(matches)}")

        if len(matches) < min_matches:
            print("Too few mutual matches, skipping pair.")
            kp_prev, des_prev = kp_curr, des_curr
            img_prev, gray_prev = img_curr, gray_curr
            continue

        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches])

        # Subpixel refine matches (improves triangulation stability)
        pts_prev_ref = refine_keypoints_subpix(gray_prev, pts_prev)
        pts_curr_ref = refine_keypoints_subpix(gray_curr, pts_curr)

        # Find essential matrix (use RANSAC); pass K so points can be in pixels
        E, maskE = cv2.findEssentialMat(pts_prev_ref, pts_curr_ref, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or maskE is None:
            print("Essential matrix failed, skipping.")
            kp_prev, des_prev = kp_curr, des_curr
            img_prev, gray_prev = img_curr, gray_curr
            continue

        maskE = maskE.ravel().astype(bool)
        if np.sum(maskE) < 8:
            print("Too few essential inliers, skipping.")
            kp_prev, des_prev = kp_curr, des_curr
            img_prev, gray_prev = img_curr, gray_curr
            continue

        pts_prev_in = pts_prev_ref[maskE]
        pts_curr_in = pts_curr_ref[maskE]

        # Recover pose
        _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts_prev_in, pts_curr_in, K)
        mask_pose = mask_pose.ravel().astype(bool)
        if np.sum(mask_pose) < 8:
            print("Too few pose inliers, skipping.")
            kp_prev, des_prev = kp_curr, des_curr
            img_prev, gray_prev = img_curr, gray_curr
            continue

        # Compose to absolute pose (correct composition)
        R_curr = (R_rel @ R_prev).astype(np.float64)
        t_curr = (R_rel @ t_prev + t_rel).astype(np.float64)

        # Use only pose inliers for triangulation
        pts_prev_final = pts_prev_in[mask_pose]
        pts_curr_final = pts_curr_in[mask_pose]

        # Undistort image points to normalized coordinates for triangulation stability
        if dist_coeffs is None:
            dist = np.zeros((5, 1), dtype=np.float64)
        else:
            dist = np.array(dist_coeffs, dtype=np.float64)
        # undistortPoints returns points in normalized (x/z, y/z) coords if P=None; but triangulatePoints expects pixel coords when P includes K.
        # We'll undistort for geometric consistency but keep triangulation with projection matrices using K included.
        pts_prev_ud = cv2.undistortPoints(pts_prev_final.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
        pts_curr_ud = cv2.undistortPoints(pts_curr_final.reshape(-1, 1, 2), K, dist).reshape(-1, 2)

        # Triangulate (use projection matrices with K included)
        P1 = K @ np.hstack((R_prev, t_prev))
        P2 = K @ np.hstack((R_curr, t_curr))
        pts4d = cv2.triangulatePoints(P1, P2, pts_prev_final.T, pts_curr_final.T)
        pts3d = (pts4d[:3] / (pts4d[3] + 1e-12)).T
        total_raw += len(pts3d)
        print(f"Triangulated {len(pts3d)} points (raw)")

        # Determine a sensible max_distance if not provided (percentile)
        cam_center_prev = (-R_prev.T @ t_prev).reshape(3)
        if max_distance is None:
            dists = np.linalg.norm(pts3d - cam_center_prev[None, :], axis=1)
            max_distance_here = float(np.percentile(dists, 98) * 1.2 + 1e-6) if len(dists) else 1e6
        else:
            max_distance_here = float(max_distance)

        # Use undistorted 2D points for reprojection checks (projected via P1/P2 include K)
        # Note: pts_prev_final / pts_curr_final are in pixel coords and already used in reprojection inside filter
        valid_mask, reasons = filter_points_by_triangulation(
            pts3d, R_prev, t_prev, R_curr, t_curr, K,
            pts_prev_final, pts_curr_final,
            max_reproj_error=max_reproj_error,
            min_angle_deg=min_angle_deg,
            max_distance=max_distance_here
        )

        kept = np.sum(valid_mask)
        print(f"Kept {kept}/{len(pts3d)} points after filtering. Reasons: {reasons}")

        if kept > 0:
            successful_pairs += 1
            pts_kept = pts3d[valid_mask]
            # Get colors from previous image projection (round, clip)
            coords = np.round(pts_prev_final[valid_mask]).astype(int)
            coords[:, 0] = np.clip(coords[:, 0], 0, img_prev.shape[1] - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, img_prev.shape[0] - 1)
            cols = img_prev[coords[:, 1], coords[:, 0]].astype(np.float32) / 255.0
            points3D_all.append(pts_kept)
            colors_all.append(cols)

        # Update for next iteration
        kp_prev, des_prev = kp_curr, des_curr
        img_prev, gray_prev = img_curr, gray_curr
        R_prev, t_prev = R_curr, t_curr
        camera_poses.append((R_curr.copy(), t_curr.copy()))

    # Merge results
    if len(points3D_all) == 0:
        print("No 3D points reconstructed.")
        return None, None, camera_poses

    points3D = np.vstack(points3D_all).astype(np.float64)
    colors = np.vstack(colors_all).astype(np.float32)

    print(f"\nReconstruction summary: pairs succeeded: {successful_pairs}, raw points total: {total_raw}, final points: {len(points3D)}")
    return points3D, colors, camera_poses


def save_ply(filename, points, colors):
    """Save RGB point cloud to ASCII PLY (clipped colors)."""
    assert points.shape[0] == colors.shape[0]
    n = points.shape[0]
    with open(filename, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {n}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(points, colors):
            r, g, b = np.clip((c * 255.0).astype(int), 0, 255)
            f.write(f'{p[0]} {p[1]} {p[2]} {r} {g} {b}\n')


def plot_and_save_cloud(points, colors, png_path='reconstruction.png'):
    """Create a 3D scatter and save as PNG."""
    if points is None or len(points) == 0:
        print("No points to plot.")
        return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    # Load calibration
    cal_file = 'calibration.npz'
    if not os.path.exists(cal_file):
        print(f"Calibration file '{cal_file}' not found. Exiting.")
        sys.exit(1)

    data = np.load(cal_file)
    if 'camera_matrix' not in data:
        print("Calibration file missing 'camera_matrix'. Exiting.")
        sys.exit(1)
    K = data['camera_matrix'].astype(np.float64)
    dist_coeffs = data.get('dist_coeffs', None)

    # Discover images
    image_glob = sorted(glob.glob('photos_coffe/*.png')) + sorted(glob.glob('photos_coffe/*.jpg')) + sorted(glob.glob('photos_coffe/*.JPEG'))
    image_files = sorted(set(image_glob))
    if len(image_files) == 0:
        print("No images found in 'photos_coffe' folder. Exiting.")
        sys.exit(1)

    print(f"Found {len(image_files)} images. Running reconstruction...")

    # Run reconstruction with robust defaults
    points3D, colors, poses = reconstruct_sequence(image_files, K,
                         max_reproj_error=0.2,   # very tight reprojection constraint
                         min_angle_deg=4.0,      # strong geometric constraint
                         max_distance=20.0,      # reject distant points
                         min_matches=25,         # require many feature correspondences
                         flann_checks=150,       # deeper FLANN search
                         ratio_thresh=0.65
    )

    if points3D is None or len(points3D) == 0:
        print("No reconstruction produced. Check capture / parameters.")
        sys.exit(1)

    # Simple statistical outlier removal
    mean = np.mean(points3D, axis=0)
    std = np.std(points3D, axis=0)
    mask_in = np.all(np.abs(points3D - mean) < 3.0 * std, axis=1)
    points3D_f = points3D[mask_in]
    colors_f = colors[mask_in]

    print(f"After statistical filter: {len(points3D_f)} points remain.")

    # Save outputs
    ply_path = 'point_cloud_robust.ply'
    save_ply(ply_path, points3D_f, colors_f)
    print(f"Saved PLY to {ply_path}")

    png_path = 'reconstruction_robust.png'
    plot_and_save_cloud(points3D_f, colors_f, png_path)
    print(f"Saved PNG visualization to {png_path}")

    # Optional interactive view with Open3D
    if _HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points3D_f)
        pcd.colors = o3d.utility.Vector3dVector(colors_f)
        o3d.visualization.draw_geometries([pcd])

    print("Done.")


if __name__ == '__main__':
    main()
