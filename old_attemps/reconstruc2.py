import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

def setup_matplotlib_for_plotting():
    """
    Setup matplotlib for plotting with proper configuration.
    Call this function before creating any plots to ensure proper rendering.
    """
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    
    # Configure platform-appropriate fonts
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", 
                                         "PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def draw_epipolar_lines(img1, img2, lines, pts1, pts2, sample_size=20):
    """
    Draw epipolar lines and corresponding points on image pairs
    
    Args:
        img1, img2: Input images
        lines: Epipolar lines from cv2.computeCorrespondEpilines
        pts1, pts2: Corresponding points in both images
        sample_size: Number of points to visualize (to avoid clutter)
    """
    h, w = img1.shape[:2]
    img1_color = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    
    # Sample points to avoid clutter
    indices = np.random.choice(len(pts1), min(sample_size, len(pts1)), replace=False)
    
    for idx in indices:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw epipolar line on img2
        line = lines[idx]
        x0, y0 = 0, int(-line[2] / line[1]) if abs(line[1]) > 1e-6 else 0
        x1, y1 = w, int(-(line[2] + line[0] * w) / line[1]) if abs(line[1]) > 1e-6 else h
        
        # Clamp coordinates
        y0 = np.clip(y0, 0, h-1)
        y1 = np.clip(y1, 0, h-1)
        
        img2_color = cv2.line(img2_color, (x0, y0), (x1, y1), color, 1)
        
        # Draw points
        pt1 = tuple(pts1[idx].astype(int))
        pt2 = tuple(pts2[idx].astype(int))
        img1_color = cv2.circle(img1_color, pt1, 5, color, -1)
        img2_color = cv2.circle(img2_color, pt2, 5, color, -1)
    
    return img1_color, img2_color


def visualize_matches(img1, img2, pts1, pts2, sample_size=50, save_path=None):
    """
    Visualize point matches between two images
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Convert to RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    
    # Create side-by-side image
    h = max(h1, h2)
    vis = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1_rgb
    vis[:h2, w1:w1+w2] = img2_rgb
    
    # Sample points
    indices = np.random.choice(len(pts1), min(sample_size, len(pts1)), replace=False)
    
    for idx in indices:
        color = tuple(np.random.randint(50, 255, 3).tolist())
        pt1 = tuple(pts1[idx].astype(int))
        pt2 = tuple((pts2[idx] + np.array([w1, 0])).astype(int))
        
        cv2.circle(vis, pt1, 3, color, -1)
        cv2.circle(vis, pt2, 3, color, -1)
        cv2.line(vis, pt1, pt2, color, 1)
    
    if save_path:
        setup_matplotlib_for_plotting()
        plt.figure(figsize=(16, 8))
        plt.imshow(vis)
        plt.title(f'Feature Matches (showing {len(indices)} of {len(pts1)} matches)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return vis


def filter_points_by_triangulation(pts3d, R1, t1, R2, t2, K, pts1, pts2, 
                                   max_reproj_error=5.0, min_angle_deg=0.5, max_distance=100.0):
    """
    Filter 3D points based on:
    1. Reprojection error
    2. Triangulation angle
    3. Positive depth in both cameras
    4. Reasonable distance from cameras
    """
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    
    valid_mask = np.ones(len(pts3d), dtype=bool)
    
    reasons = {'depth': 0, 'reproj': 0, 'angle': 0, 'distance': 0}
    
    # Check positive depth
    pts3d_h = np.hstack((pts3d, np.ones((len(pts3d), 1))))
    
    # In camera 1
    pts_cam1 = (R1 @ pts3d.T + t1).T
    depth1_check = pts_cam1[:, 2] > 0
    valid_mask &= depth1_check
    reasons['depth'] += np.sum(~depth1_check)
    
    # In camera 2
    pts_cam2 = (R2 @ pts3d.T + t2).T
    depth2_check = pts_cam2[:, 2] > 0
    valid_mask &= depth2_check
    reasons['depth'] += np.sum(~depth2_check)
    
    # Distance check (remove points too far from cameras)
    cam1_center = -R1.T @ t1
    cam2_center = -R2.T @ t2
    dist1 = np.linalg.norm(pts3d - cam1_center.T, axis=1)
    dist2 = np.linalg.norm(pts3d - cam2_center.T, axis=1)
    dist_check = (dist1 < max_distance) & (dist2 < max_distance)
    reasons['distance'] += np.sum(~dist_check)
    valid_mask &= dist_check
    
    # Reprojection error
    pts_proj1 = (P1 @ pts3d_h.T).T
    pts_proj1 = pts_proj1[:, :2] / pts_proj1[:, 2:3]
    error1 = np.linalg.norm(pts_proj1 - pts1, axis=1)
    
    pts_proj2 = (P2 @ pts3d_h.T).T
    pts_proj2 = pts_proj2[:, :2] / pts_proj2[:, 2:3]
    error2 = np.linalg.norm(pts_proj2 - pts2, axis=1)
    
    reproj_check = (error1 < max_reproj_error) & (error2 < max_reproj_error)
    reasons['reproj'] += np.sum(~reproj_check)
    valid_mask &= reproj_check
    
    # Triangulation angle
    for i in range(len(pts3d)):
        if not valid_mask[i]:
            continue
        ray1 = pts3d[i] - cam1_center.flatten()
        ray2 = pts3d[i] - cam2_center.flatten()
        ray1_norm = ray1 / (np.linalg.norm(ray1) + 1e-10)
        ray2_norm = ray2 / (np.linalg.norm(ray2) + 1e-10)
        
        cos_angle = np.clip(np.dot(ray1_norm, ray2_norm), -1, 1)
        angle_deg = np.degrees(np.arccos(cos_angle))
        
        if angle_deg < min_angle_deg or angle_deg > 180 - min_angle_deg:
            valid_mask[i] = False
            reasons['angle'] += 1
    
    return valid_mask, reasons


def main():
    setup_matplotlib_for_plotting()
    
    # Load calibration
    print("Loading calibration data...")
    data = np.load('calibration.npz')
    K = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    
    print(f"Camera matrix:\n{K}")
    print(f"Distortion coefficients: {dist_coeffs.flatten()}")
    
    # Load and sort images
    image_files = sorted(glob.glob('image_Luigi/png/*.png'))
    if len(image_files) == 0:
        image_files = sorted(glob.glob('image_Luigi/png/*.jpg'))
    
    print(f"\nTotal images found: {len(image_files)}")
    
    if len(image_files) == 0:
        print("ERROR: No images found!")
        return
    
    # SIFT detector and matcher
    sift = cv2.SIFT_create(nfeatures=3000)
    
    # Use FLANN matcher for better performance
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Storage
    camera_poses = []
    points3D_all = []
    colors_all = []
    
    # Initialize first camera at origin
    R_prev = np.eye(3)
    t_prev = np.zeros((3, 1))
    camera_poses.append((R_prev.copy(), t_prev.copy()))
    
    # Read first image
    img_prev = cv2.imread(image_files[0])
    gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    kp_prev, des_prev = sift.detectAndCompute(gray_prev, None)
    
    print(f"\nImage 0: {len(kp_prev)} keypoints detected")
    
    # Process image pairs
    for i in range(1, min(len(image_files), 10)):  # Limit to first 10 for debugging
        print(f"\n{'='*60}")
        print(f"Processing image pair: {i-1} -> {i}")
        print(f"{'='*60}")
        
        # Read current image
        img_curr = cv2.imread(image_files[i])
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
        kp_curr, des_curr = sift.detectAndCompute(gray_curr, None)
        
        print(f"Image {i}: {len(kp_curr)} keypoints detected")
        
        # Match features with ratio test
        matches = flann.knnMatch(des_prev, des_curr, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:  # More lenient threshold
                    good_matches.append(m)
        
        print(f"Good matches after ratio test: {len(good_matches)}")
        
        if len(good_matches) < 30:
            print(f"WARNING: Too few matches ({len(good_matches)}), skipping this pair")
            kp_prev, des_prev = kp_curr, des_curr
            img_prev = img_curr
            gray_prev = gray_curr
            continue
        
        # Extract matched points
        pts_prev = np.array([kp_prev[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        pts_curr = np.array([kp_curr[m.trainIdx].pt for m in good_matches], dtype=np.float32)
        
        # Find Essential matrix using normalized coordinates
        E, mask_E = cv2.findEssentialMat(pts_prev, pts_curr, K, 
                                          method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None or mask_E is None:
            print("ERROR: Essential matrix computation failed, skipping")
            kp_prev, des_prev = kp_curr, des_curr
            img_prev = img_curr
            gray_prev = gray_curr
            continue
        
        inliers = mask_E.ravel() == 1
        print(f"Essential matrix inliers: {np.sum(inliers)}/{len(pts_prev)}")
        
        if np.sum(inliers) < 20:
            print("WARNING: Too few inliers, skipping")
            kp_prev, des_prev = kp_curr, des_curr
            img_prev = img_curr
            gray_prev = gray_curr
            continue
        
        # Filter to inliers only
        pts_prev_inliers = pts_prev[inliers]
        pts_curr_inliers = pts_curr[inliers]
        
        # Recover pose
        _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts_prev_inliers, pts_curr_inliers, K)
        
        pose_inliers = mask_pose.ravel() > 0
        print(f"Pose recovery inliers: {np.sum(pose_inliers)}/{len(pts_prev_inliers)}")
        
        # Filter to pose inliers
        pts_prev_final = pts_prev_inliers[pose_inliers]
        pts_curr_final = pts_curr_inliers[pose_inliers]
        
        # Compute absolute pose - DON'T normalize translation to preserve scale
        R_curr = R_rel @ R_prev
        t_curr = R_prev @ t_rel + t_prev
        
        baseline = np.linalg.norm(t_rel)
        print(f"Relative translation magnitude: {baseline:.4f}")
        print(f"Camera {i} position: {t_curr.T}")
        
        camera_poses.append((R_curr.copy(), t_curr.copy()))
        
        # Triangulate points
        P1 = K @ np.hstack((R_prev, t_prev))
        P2 = K @ np.hstack((R_curr, t_curr))
        
        pts4d_h = cv2.triangulatePoints(P1, P2, pts_prev_final.T, pts_curr_final.T)
        pts3d = (pts4d_h[:3] / pts4d_h[3]).T
        
        print(f"Triangulated {len(pts3d)} points")
        print(f"  Depth range in cam1: [{(R_prev @ pts3d.T + t_prev)[2].min():.2f}, {(R_prev @ pts3d.T + t_prev)[2].max():.2f}]")
        print(f"  Depth range in cam2: [{(R_curr @ pts3d.T + t_curr)[2].min():.2f}, {(R_curr @ pts3d.T + t_curr)[2].max():.2f}]")
        
        # Filter points by quality with more lenient thresholds
        valid_mask, reasons = filter_points_by_triangulation(
            pts3d, R_prev, t_prev, R_curr, t_curr, K, 
            pts_prev_final, pts_curr_final,
            max_reproj_error=5.0,  # More lenient
            min_angle_deg=0.5,     # More lenient
            max_distance=100.0     # Maximum distance from camera
        )
        
        pts3d_filtered = pts3d[valid_mask]
        print(f"3D points after filtering: {len(pts3d_filtered)}/{len(pts3d)}")
        print(f"  Filtering reasons - depth: {reasons['depth']}, reproj: {reasons['reproj']}, angle: {reasons['angle']}, distance: {reasons['distance']}")
        
        if len(pts3d_filtered) > 0:
            points3D_all.append(pts3d_filtered)
            
            # Get colors from original image
            pts_for_color = pts_prev_final[valid_mask].astype(int)
            # Ensure indices are within bounds
            pts_for_color[:, 0] = np.clip(pts_for_color[:, 0], 0, img_prev.shape[1]-1)
            pts_for_color[:, 1] = np.clip(pts_for_color[:, 1], 0, img_prev.shape[0]-1)
            colors = img_prev[pts_for_color[:, 1], pts_for_color[:, 0]]
            colors_all.append(colors / 255.0)  # Normalize to [0, 1]
        
        # ============ DEBUGGING VISUALIZATIONS ============
        
        # 1. Visualize matches
        print(f"\nSaving match visualization...")
        vis_matches = visualize_matches(
            gray_prev, gray_curr, pts_prev_final, pts_curr_final,
            sample_size=50,
            save_path=f'debug_matches_{i-1}_to_{i}.png'
        )
        
        # 2. Visualize epipolar lines
        print(f"Computing epipolar lines...")
        # Compute fundamental matrix for visualization
        F, mask_F = cv2.findFundamentalMat(pts_prev_final, pts_curr_final, cv2.FM_RANSAC, 1.0, 0.99)
        
        if F is not None and F.shape == (3, 3):
            # Compute epipolar lines
            lines_in_curr = cv2.computeCorrespondEpilines(pts_prev_final.reshape(-1, 1, 2), 1, F)
            lines_in_curr = lines_in_curr.reshape(-1, 3)
            
            lines_in_prev = cv2.computeCorrespondEpilines(pts_curr_final.reshape(-1, 1, 2), 2, F)
            lines_in_prev = lines_in_prev.reshape(-1, 3)
            
            # Draw epipolar lines
            img_prev_epi, img_curr_epi = draw_epipolar_lines(
                gray_prev, gray_curr, lines_in_curr, pts_prev_final, pts_curr_final, sample_size=15
            )
            
            # Save epipolar visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            axes[0].imshow(img_prev_epi)
            axes[0].set_title(f'Image {i-1} with matched points')
            axes[0].axis('off')
            
            axes[1].imshow(img_curr_epi)
            axes[1].set_title(f'Image {i} with epipolar lines')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'debug_epipolar_{i-1}_to_{i}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved epipolar line visualization")
        else:
            print("WARNING: Could not compute fundamental matrix for epipolar visualization")
        
        # Update for next iteration
        kp_prev, des_prev = kp_curr, des_curr
        img_prev = img_curr
        gray_prev = gray_curr
        R_prev, t_prev = R_curr, t_curr
    
    # ============ FINAL 3D RECONSTRUCTION VISUALIZATION ============
    
    if len(points3D_all) == 0:
        print("\nERROR: No 3D points reconstructed!")
        print("This could be due to:")
        print("  1. Insufficient baseline between cameras")
        print("  2. Poor image quality or textureless regions")
        print("  3. Incorrect camera calibration")
        print("  4. Too strict filtering parameters")
        return
    
    points3D_all = np.concatenate(points3D_all, axis=0)
    colors_all = np.concatenate(colors_all, axis=0)
    
    print(f"\n{'='*60}")
    print(f"FINAL RECONSTRUCTION STATISTICS")
    print(f"{'='*60}")
    print(f"Total 3D points: {len(points3D_all)}")
    print(f"Total cameras: {len(camera_poses)}")
    print(f"Point cloud extent:")
    print(f"  X: [{points3D_all[:,0].min():.2f}, {points3D_all[:,0].max():.2f}]")
    print(f"  Y: [{points3D_all[:,1].min():.2f}, {points3D_all[:,1].max():.2f}]")
    print(f"  Z: [{points3D_all[:,2].min():.2f}, {points3D_all[:,2].max():.2f}]")
    
    # Remove outliers (statistical filtering)
    mean = np.mean(points3D_all, axis=0)
    std = np.std(points3D_all, axis=0)
    dist_from_mean = np.abs(points3D_all - mean)
    outlier_mask = np.all(dist_from_mean < 3 * std, axis=1)
    
    points3D_filtered = points3D_all[outlier_mask]
    colors_filtered = colors_all[outlier_mask]
    
    print(f"Points after outlier removal: {len(points3D_filtered)}/{len(points3D_all)}")
    
    # Create 3D visualization
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points with colors
    ax.scatter(points3D_filtered[:, 0], 
               points3D_filtered[:, 1], 
               points3D_filtered[:, 2],
               c=colors_filtered, s=1, alpha=0.5)
    
    # Plot camera positions
    cam_positions = []
    for idx, (R, t) in enumerate(camera_poses):
        cam_center = (-R.T @ t).flatten()
        # Ensure it's a proper 1D array with shape (3,)
        cam_center = np.array([cam_center[0], cam_center[1], cam_center[2]])
        cam_positions.append(cam_center)
        # Convert to Python floats to avoid matplotlib warnings
        x, y, z = float(cam_center[0]), float(cam_center[1]), float(cam_center[2])
        ax.scatter(x, y, z, c='red', marker='o', s=100, edgecolors='black', linewidths=2)
        ax.text(x, y, z, f'  C{idx}', fontsize=10)
    
    # Plot camera trajectory
    if len(cam_positions) > 1:
        cam_positions = np.vstack(cam_positions)  # Ensure proper 2D array (n_cameras, 3)
        ax.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], 
                'r-', linewidth=2, label='Camera trajectory')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'3D Reconstruction: {len(points3D_filtered)} points, {len(camera_poses)} cameras', 
                 fontsize=14)
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([
        points3D_filtered[:, 0].max() - points3D_filtered[:, 0].min(),
        points3D_filtered[:, 1].max() - points3D_filtered[:, 1].min(),
        points3D_filtered[:, 2].max() - points3D_filtered[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points3D_filtered[:, 0].max() + points3D_filtered[:, 0].min()) * 0.5
    mid_y = (points3D_filtered[:, 1].max() + points3D_filtered[:, 1].min()) * 0.5
    mid_z = (points3D_filtered[:, 2].max() + points3D_filtered[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig('reconstruction_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n3D reconstruction saved to 'reconstruction_3d.png'")
    
    # Save point cloud to PLY format
    with open('point_cloud.ply', 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points3D_filtered)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        
        for point, color in zip(points3D_filtered, colors_filtered):
            r, g, b = (color * 255).astype(int)
            f.write(f'{point[0]} {point[1]} {point[2]} {r} {g} {b}\n')
    
    print(f"Point cloud saved to 'point_cloud.ply'")
    print("\nReconstruction complete!")


if __name__ == '__main__':
    main()
