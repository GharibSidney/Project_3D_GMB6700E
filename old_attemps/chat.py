#!/usr/bin/env python3
"""
Robust reconstruction script with explicit global pose chaining (first camera = world origin).

Changes:
 - Ensures all camera poses are chained into a single global frame.
 - Triangulated points are produced/confirmed in that global frame before accumulation.
 - Accumulate all points into a global list (all_pts3d) and run a final strict cleaning pass
   (depth percentile + spatial percentile) to remove floating islands / duplicated clusters.
"""

import glob
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False
    print("Warning: open3d not available - interactive view disabled.")


def filter_points_by_triangulation(pts3d, R1, t1, R2, t2, K, pts1, pts2,
                                   max_reproj_error=2.0, min_angle_deg=0.3, max_distance=200.0):
    """Filter 3D points with threshold checks (safe divisions and guards)."""
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

    mutual = []
    for a_idx, b_idx in good_ab.items():
        if b_idx in good_ba and good_ba[b_idx] == a_idx:
            m = cv2.DMatch(_queryIdx=a_idx, _trainIdx=b_idx, _imgIdx=0, _distance=0.0)
            mutual.append(m)
    return mutual


def refine_keypoints_subpix(gray_img, pts, win_size=(5,5), zero_zone=(-1,-1), criteria=None):
    if criteria is None:
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 1e-4)
    if len(pts) == 0:
        return pts
    pts_cv = pts.reshape(-1, 1, 2).astype(np.float32)
    cv2.cornerSubPix(gray_img, pts_cv, win_size, zero_zone, criteria)
    return pts_cv.reshape(-1, 2)


def reconstruct_sequence(image_files, K,
                         dist_coeffs=None,
                         max_reproj_error=0.8, min_angle_deg=1.0, max_distance=None,
                         min_matches=12, flann_checks=64, ratio_thresh=0.75):
    """
    Reconstruct pairwise along the image sequence in a single global frame
    (first camera = world origin).
    Returns:
      points3D (Nx3), colors (Nx3), camera_poses list [(R,t), ...], observations list
    observations: list of tuples (cam_idx, point_global_idx, u, v)
    where each point has observations from the two views that triangulated it.
    """
    # ... (keep the same setup as before: SIFT/FLANN, initialization) ...

    # Copy the same setup code from your previous reconstruct_sequence up to the loop:
    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=flann_checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    img_prev = cv2.imread(image_files[0])
    if img_prev is None:
        raise FileNotFoundError(f"Can't read image {image_files[0]}")
    gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    kp_prev, des_prev = sift.detectAndCompute(gray_prev, None)

    R_global_prev = np.eye(3, dtype=np.float64)
    t_global_prev = np.zeros((3, 1), dtype=np.float64)
    camera_poses = [(R_global_prev.copy(), t_global_prev.copy())]

    all_pts3d = []
    all_colors = []
    observations = []  # list of (cam_idx, point_global_idx, u, v)

    total_raw = 0
    successful_pairs = 0
    point_global_idx = 0

    for i in range(1, len(image_files)):
        print(f"\n--- Pair {i-1} -> {i} ---")
        img_curr = cv2.imread(image_files[i])
        if img_curr is None:
            print(f"Warning: Can't read {image_files[i]}, skipping.")
            continue
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
        kp_curr, des_curr = sift.detectAndCompute(gray_curr, None)

        if des_prev is None or des_curr is None or len(kp_prev) == 0 or len(kp_curr) == 0:
            print("No descriptors in one image; skipping.")
            kp_prev, des_prev = kp_curr, des_curr
            img_prev, gray_prev = img_curr, gray_curr
            continue

        matches = mutual_matches(des_prev, des_curr, flann, ratio_thresh=ratio_thresh)
        print(f"Mutual matches: {len(matches)}")
        if len(matches) < min_matches:
            print("Too few mutual matches, skipping pair.")
            kp_prev, des_prev = kp_curr, des_curr
            img_prev, gray_prev = img_curr, gray_curr
            continue

        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches])

        pts_prev_ref = refine_keypoints_subpix(gray_prev, pts_prev)
        pts_curr_ref = refine_keypoints_subpix(gray_curr, pts_curr)

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

        _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts_prev_in, pts_curr_in, K)
        mask_pose = mask_pose.ravel().astype(bool)
        if np.sum(mask_pose) < 8:
            print("Too few pose inliers, skipping.")
            kp_prev, des_prev = kp_curr, des_curr
            img_prev, gray_prev = img_curr, gray_curr
            continue

        R_global_curr = (R_rel @ R_global_prev).astype(np.float64)
        t_global_curr = (R_rel @ t_global_prev + t_rel).astype(np.float64)

        pts_prev_final = pts_prev_in[mask_pose]
        pts_curr_final = pts_curr_in[mask_pose]

        P1 = K @ np.hstack((R_global_prev, t_global_prev))
        P2 = K @ np.hstack((R_global_curr, t_global_curr))
        pts4d = cv2.triangulatePoints(P1, P2, pts_prev_final.T, pts_curr_final.T)
        pts3d = (pts4d[:3] / (pts4d[3] + 1e-12)).T
        total_raw += len(pts3d)
        print(f"Triangulated {len(pts3d)} points (raw) in global frame")

        cam_center_prev = (-R_global_prev.T @ t_global_prev).reshape(3)
        if max_distance is None:
            dists = np.linalg.norm(pts3d - cam_center_prev[None, :], axis=1)
            max_distance_here = float(np.percentile(dists, 98) * 1.2 + 1e-6) if len(dists) else 1e6
        else:
            max_distance_here = float(max_distance)

        valid_mask, reasons = filter_points_by_triangulation(
            pts3d, R_global_prev, t_global_prev, R_global_curr, t_global_curr, K,
            pts_prev_final, pts_curr_final,
            max_reproj_error=max_reproj_error,
            min_angle_deg=min_angle_deg,
            max_distance=max_distance_here
        )

        kept = np.sum(valid_mask)
        print(f"Kept {kept}/{len(pts3d)} points after filtering. Reasons: {reasons}")

        if kept > 0:
            successful_pairs += 1
            pts_kept_world = pts3d[valid_mask]
            coords = np.round(pts_prev_final[valid_mask]).astype(int)
            coords[:, 0] = np.clip(coords[:, 0], 0, img_prev.shape[1] - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, img_prev.shape[0] - 1)
            cols = img_prev[coords[:, 1], coords[:, 0]].astype(np.float32) / 255.0

            # Append points and colors
            all_pts3d.append(pts_kept_world)
            all_colors.append(cols)

            # Add observations: for each new global point, record two observations
            for k_idx, uv_prev, uv_curr in zip(range(len(pts_kept_world)),
                                               pts_prev_final[valid_mask],
                                               pts_curr_final[valid_mask]):
                # global index of this point
                gidx = point_global_idx
                # prev camera index is len(camera_poses)-1 (since camera_poses stores prev camera)
                cam_prev_idx = len(camera_poses) - 1
                cam_curr_idx = cam_prev_idx + 1
                u_prev, v_prev = float(uv_prev[0]), float(uv_prev[1])
                u_curr, v_curr = float(uv_curr[0]), float(uv_curr[1])
                observations.append((cam_prev_idx, gidx, u_prev, v_prev))
                observations.append((cam_curr_idx, gidx, u_curr, v_curr))
                point_global_idx += 1

        # Update for next iteration
        kp_prev, des_prev = kp_curr, des_curr
        img_prev, gray_prev = img_curr, gray_curr
        R_global_prev, t_global_prev = R_global_curr, t_global_curr
        camera_poses.append((R_global_curr.copy(), t_global_curr.copy()))

    # Merge results
    if len(all_pts3d) == 0:
        print("No 3D points reconstructed.")
        return None, None, camera_poses, observations

    points3D = np.vstack(all_pts3d).astype(np.float64)
    colors = np.vstack(all_colors).astype(np.float32)

    print(f"\nReconstruction summary: pairs succeeded: {successful_pairs}, raw points total: {total_raw}, final points: {len(points3D)}")
    return points3D, colors, camera_poses, observations


def bundle_adjustment(camera_poses, points3D, observations, K, fix_first_camera=True, verbose=2, max_nfev=200):
    """
    Bundle adjust camera poses and 3D points using scipy.least_squares.
    camera_poses: list of (R, t) in global frame
    points3D: Nx3 array
    observations: list of (cam_idx, point_idx, u, v)
    K: camera intrinsics 3x3
    Returns: refined_camera_poses, refined_points3D
    """

    n_cams = len(camera_poses)
    n_pts = points3D.shape[0]

    # Build index lists for observations to speed residual eval
    obs_cam_idx = np.array([o[0] for o in observations], dtype=int)
    obs_pt_idx = np.array([o[1] for o in observations], dtype=int)
    obs_uv = np.array([[o[2], o[3]] for o in observations], dtype=np.float64)

    # Pack initial parameters:
    # For cameras: rvec (3) and tvec (3) per camera. For points: XYZ (3)
    def rodrigues_from_R(R):
        rvec, _ = cv2.Rodrigues(R)
        return rvec.flatten()

    cam_r0 = np.vstack([rodrigues_from_R(R) for (R, t) in camera_poses])  # (n_cams,3)
    cam_t0 = np.vstack([t.flatten() for (R, t) in camera_poses])           # (n_cams,3)
    pts0 = points3D.copy()  # (n_pts,3)

    # If fix_first_camera: do not include first camera in parameter vector
    cam_var_idx = np.arange(n_cams)  # indices into camera arrays
    if fix_first_camera:
        var_cam_idx = cam_var_idx[1:]  # cameras to optimize
    else:
        var_cam_idx = cam_var_idx

    # Build parameter vector
    def pack_params(cam_r, cam_t, pts):
        # only pack variable cameras
        cam_params = []
        for ci in var_cam_idx:
            cam_params.append(cam_r[ci])
            cam_params.append(cam_t[ci])
        cam_params = np.hstack(cam_params).ravel()
        pts_params = pts.ravel()
        return np.hstack([cam_params, pts_params])

    def unpack_params(x):
        # given x, fill cam_r_opt, cam_t_opt, pts_opt
        cam_r_opt = cam_r0.copy()
        cam_t_opt = cam_t0.copy()
        offset = 0
        for ci in var_cam_idx:
            cam_r_opt[ci] = x[offset:offset+3]; offset += 3
            cam_t_opt[ci] = x[offset:offset+3]; offset += 3
        pts_opt = x[offset:].reshape((-1, 3))
        return cam_r_opt, cam_t_opt, pts_opt

    x0 = pack_params(cam_r0, cam_t0, pts0)

    # residual function
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])

    def residuals(x):
        cam_r_opt, cam_t_opt, pts_opt = unpack_params(x)
        res = np.empty((obs_uv.shape[0] * 2,), dtype=np.float64)

        for ci in np.unique(obs_cam_idx):
            # indices of observations for this camera
            mask = (obs_cam_idx == ci)
            pts_ids = obs_pt_idx[mask]
            uv_obs = obs_uv[mask]
            rvec = cam_r_opt[ci].reshape(3, 1)
            tvec = cam_t_opt[ci].reshape(3, 1)

            # project all visible points at once
            X = pts_opt[pts_ids].reshape(-1, 1, 3)
            xproj, _ = cv2.projectPoints(X, rvec, tvec, K, None)
            uv_proj = xproj.reshape(-1, 2)

            # residuals for this camera’s points
            diff = (uv_proj - uv_obs).ravel()
            res[2*np.where(mask)[0][0]:2*np.where(mask)[0][0]+diff.size] = diff

        return res

    print("Starting bundle adjustment: params size =", x0.size, "residuals =", obs_uv.shape[0]*2)
    # run least squares
    result = least_squares(residuals, x0, verbose=verbose, max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8, method='trf')

    cam_r_opt, cam_t_opt, pts_opt = unpack_params(result.x)

    # convert optimized cam r/t back to R,t
    refined_poses = []
    for i in range(n_cams):
        rvec = cam_r_opt[i].reshape(3, 1)
        R_i, _ = cv2.Rodrigues(rvec)
        t_i = cam_t_opt[i].reshape(3, 1)
        refined_poses.append((R_i, t_i))

    return refined_poses, pts_opt

def save_ply(filename, points, colors):
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

    image_glob = sorted(glob.glob('photos_coffe/*.png')) + sorted(glob.glob('photos_coffe/*.jpg')) + sorted(glob.glob('photos_coffe/*.JPEG'))
    image_files = sorted(set(image_glob))
    if len(image_files) == 0:
        print("No images found in 'photos_coffe' folder. Exiting.")
        sys.exit(1)

    print(f"Found {len(image_files)} images. Running reconstruction...")

    # Strict (high-quality) parameters (you asked for strictest)
    points3D, colors, poses, observations  = reconstruct_sequence(
        image_files, K,
        dist_coeffs=dist_coeffs,
        max_reproj_error=0.2,
        min_angle_deg=4.0,
        max_distance=20.0,
        min_matches=25,
        flann_checks=150,
        ratio_thresh=0.65
    )

    print("Running bundle adjustment (this may take a while)...")
    refined_poses, refined_points = bundle_adjustment(poses, points3D, observations, K, fix_first_camera=True, verbose=2, max_nfev=200)
    # update points3D and poses with refined results
    points3D = refined_points
    poses = refined_poses

    if points3D is None or len(points3D) == 0:
        print("No reconstruction produced. Check capture / parameters.")
        sys.exit(1)


    # === Final global cleaning (depth + spatial percentiles) ===
    # Remove extreme depth outliers (keep central 96%)
    z_vals = points3D[:, 2]
    lo_z, hi_z = np.percentile(z_vals, 2), np.percentile(z_vals, 98)
    depth_mask = (z_vals > lo_z) & (z_vals < hi_z)

    # Remove spatial outliers (keep central 95% by distance to mean)
    dist_from_mean = np.linalg.norm(points3D - np.mean(points3D, axis=0), axis=1)
    spatial_thresh = np.percentile(dist_from_mean, 95)
    spatial_mask = dist_from_mean < spatial_thresh

    strict_mask = depth_mask & spatial_mask
    pts3d_clean = points3D[strict_mask]
    colors_clean = colors[strict_mask]

    print(f"After final cleaning: {len(pts3d_clean)} points remain (from {len(points3D)})")

    ply_path = 'point_cloud_strict_global.ply'
    save_ply(ply_path, pts3d_clean, colors_clean)
    print(f"Saved PLY to {ply_path}")

    png_path = 'reconstruction_strict_global.png'
    plot_and_save_cloud(pts3d_clean, colors_clean, png_path)
    print(f"Saved PNG visualization to {png_path}")

    if _HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts3d_clean)
        pcd.colors = o3d.utility.Vector3dVector(colors_clean)
        o3d.visualization.draw_geometries([pcd])

    print("Done.")


if __name__ == '__main__':
    main()
