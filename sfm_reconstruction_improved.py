import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

def setup_matplotlib_for_plotting():
    """Setup matplotlib for plotting with proper configuration."""
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", 
                                         "PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def visualize_matches(img1, img2, pts1, pts2, sample_size=50, save_path=None):
    """Visualize point matches between two images"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    
    h = max(h1, h2)
    vis = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1_rgb
    vis[:h2, w1:w1+w2] = img2_rgb
    
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
                                   max_reproj_error=2.0, min_angle_deg=0.3, max_distance=5.0):
    """Filter 3D points with very lenient thresholds"""
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    
    valid_mask = np.ones(len(pts3d), dtype=bool)
    reasons = {'depth': 0, 'reproj': 0, 'angle': 0, 'distance': 0}
    
    pts3d_h = np.hstack((pts3d, np.ones((len(pts3d), 1))))
    
    # Check positive depth
    pts_cam1 = (R1 @ pts3d.T + t1).T
    depth1_check = pts_cam1[:, 2] > 0
    valid_mask &= depth1_check
    reasons['depth'] += np.sum(~depth1_check)
    
    pts_cam2 = (R2 @ pts3d.T + t2).T
    depth2_check = pts_cam2[:, 2] > 0
    valid_mask &= depth2_check
    reasons['depth'] += np.sum(~depth2_check)
    
    # Distance check
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
    
    # Triangulation angle (very lenient)
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
    
    print("="*70)
    print("IMPROVED 3D RECONSTRUCTION WITH BETTER PARAMETERS")
    print("="*70)
    
    # Load calibration
    print("\nLoading calibration data...")
    data = np.load('calibration.npz')
    K = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    
    print(f"Camera matrix:\n{K}")
    
    # Load ALL images
    image_files = sorted(glob.glob('photos_coffe/*.png'))
    if len(image_files) == 0:
        image_files = sorted(glob.glob('photos_coffe/*.jpg'))
    
    if len(image_files) == 0:
        image_files = sorted(glob.glob('photos_coffe/*.JPEG'))
    
    print(f"\nTotal images found: {len(image_files)}")
    print(f"Processing ALL {len(image_files)} images for better reconstruction")
    
    if len(image_files) == 0:
        print("ERROR: No images found!")
        return
    
    # Improved SIFT with MORE features
    print("\nInitializing feature detector with 5000 keypoints...")
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.03, edgeThreshold=8)
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)  # More checks for better matching
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Storage
    camera_poses = []
    points3D_all = []
    colors_all = []
    
    # Initialize first camera
    R_prev = np.eye(3)
    t_prev = np.zeros((3, 1))
    camera_poses.append((R_prev.copy(), t_prev.copy()))
    
    # Read first image
    img_prev = cv2.imread(image_files[0])
    gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    kp_prev, des_prev = sift.detectAndCompute(gray_prev, None)
    
    print(f"\nImage 0: {len(kp_prev)} keypoints detected")
    
    successful_pairs = 0
    total_3d_points_raw = 0
    
    # Process ALL image pairs
    for i in range(1, len(image_files)):
        print(f"\n{'='*70}")
        print(f"Processing image pair: {i-1} -> {i}")
        print(f"{'='*70}")
        
        # Read current image
        img_curr = cv2.imread(image_files[i])
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
        kp_curr, des_curr = sift.detectAndCompute(gray_curr, None)
        
        print(f"Image {i}: {len(kp_curr)} keypoints detected")
        
        # Match with MORE lenient ratio test
        matches = flann.knnMatch(des_prev, des_curr, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.8 * n.distance:  # Very lenient
                    good_matches.append(m)
        
        print(f"Good matches after ratio test: {len(good_matches)}")
        
        if len(good_matches) < 20:  # Lower threshold
            print(f"WARNING: Too few matches ({len(good_matches)}), skipping")
            kp_prev, des_prev = kp_curr, des_curr
            img_prev = img_curr
            gray_prev = gray_curr
            continue
        
        # Extract matched points
        pts_prev = np.array([kp_prev[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        pts_curr = np.array([kp_curr[m.trainIdx].pt for m in good_matches], dtype=np.float32)
        
        # Essential matrix with more lenient threshold
        E, mask_E = cv2.findEssentialMat(pts_prev, pts_curr, K, 
                                          method=cv2.RANSAC, prob=0.999, threshold=2.0)
        
        if E is None or mask_E is None:
            print("ERROR: Essential matrix failed, skipping")
            kp_prev, des_prev = kp_curr, des_curr
            img_prev = img_curr
            gray_prev = gray_curr
            continue
        
        inliers = mask_E.ravel() == 1
        print(f"Essential matrix inliers: {np.sum(inliers)}/{len(pts_prev)} ({100*np.sum(inliers)/len(pts_prev):.1f}%)")
        
        if np.sum(inliers) < 15:  # Lower threshold
            print("WARNING: Too few inliers, skipping")
            kp_prev, des_prev = kp_curr, des_curr
            img_prev = img_curr
            gray_prev = gray_curr
            continue
        
        # Filter to inliers
        pts_prev_inliers = pts_prev[inliers]
        pts_curr_inliers = pts_curr[inliers]
        
        # Recover pose
        _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts_prev_inliers, pts_curr_inliers, K)
        
        pose_inliers = mask_pose.ravel() > 0
        print(f"Pose recovery inliers: {np.sum(pose_inliers)}/{len(pts_prev_inliers)}")
        
        pts_prev_final = pts_prev_inliers[pose_inliers]
        pts_curr_final = pts_curr_inliers[pose_inliers]
        
        # Compute absolute pose
        R_curr = R_rel @ R_prev
        t_curr = R_prev @ t_rel + t_prev
        
        baseline = np.linalg.norm(t_rel)
        print(f"Relative baseline: {baseline:.4f}")
        print(f"Camera {i} position: {t_curr.flatten()}")
        
        camera_poses.append((R_curr.copy(), t_curr.copy()))
        
        # Triangulate
        P1 = K @ np.hstack((R_prev, t_prev))
        P2 = K @ np.hstack((R_curr, t_curr))
        
        pts4d_h = cv2.triangulatePoints(P1, P2, pts_prev_final.T, pts_curr_final.T)
        pts3d = (pts4d_h[:3] / pts4d_h[3]).T
        
        total_3d_points_raw += len(pts3d)
        
        print(f"Triangulated {len(pts3d)} points")
        
        valid_mask, reasons = filter_points_by_triangulation(
            pts3d, R_prev, t_prev, R_curr, t_curr, K, 
            pts_prev_final, pts_curr_final,
            max_reproj_error=0.8,
            min_angle_deg=2.0,   
            max_distance=50.0
        )
        
        pts3d_filtered = pts3d[valid_mask]
        print(f"Points after filtering: {len(pts3d_filtered)}/{len(pts3d)} ({100*len(pts3d_filtered)/len(pts3d):.1f}%)")
        print(f"  Rejected - depth: {reasons['depth']}, reproj: {reasons['reproj']}, angle: {reasons['angle']}, distance: {reasons['distance']}")
        
        if len(pts3d_filtered) > 0:
            successful_pairs += 1
            points3D_all.append(pts3d_filtered)
            
            # Get colors
            pts_for_color = pts_prev_final[valid_mask].astype(int)
            pts_for_color[:, 0] = np.clip(pts_for_color[:, 0], 0, img_prev.shape[1]-1)
            pts_for_color[:, 1] = np.clip(pts_for_color[:, 1], 0, img_prev.shape[0]-1)
            colors = img_prev[pts_for_color[:, 1], pts_for_color[:, 0]]
            colors_all.append(colors / 255.0)
            
            # Save match visualization for first few pairs
            if i <= 5:
                visualize_matches(
                    gray_prev, gray_curr, pts_prev_final, pts_curr_final,
                    sample_size=100,
                    save_path=f'debug_matches_{i-1}_to_{i}.png'
                )
                print(f"  Saved match visualization")
        
        # Update for next iteration
        kp_prev, des_prev = kp_curr, des_curr
        img_prev = img_curr
        gray_prev = gray_curr
        R_prev, t_prev = R_curr, t_curr
    
    # ============ FINAL RECONSTRUCTION ============
    
    print(f"\n{'='*70}")
    print(f"RECONSTRUCTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Successful image pairs: {successful_pairs}/{len(image_files)-1}")
    print(f"Total 3D points (raw): {total_3d_points_raw}")
    
    if len(points3D_all) == 0:
        print("\nERROR: No 3D points reconstructed!")
        print("\nPossible issues:")
        print("  1. Images have too little overlap")
        print("  2. Camera moved too little between shots")
        print("  3. Scene lacks texture/features")
        print("  4. Lighting conditions vary too much")
        print("\nSuggestions:")
        print("  - Take more photos with 60-80% overlap")
        print("  - Move camera in a circular pattern around object")
        print("  - Ensure good lighting")
        print("  - Use a textured object or add markers")
        return
    
    points3D_all = np.concatenate(points3D_all, axis=0)
    colors_all = np.concatenate(colors_all, axis=0)
    
    print(f"Total 3D points (after filtering): {len(points3D_all)}")
    print(f"Total cameras: {len(camera_poses)}")
    print(f"\nPoint cloud extent:")
    print(f"  X: [{points3D_all[:,0].min():.2f}, {points3D_all[:,0].max():.2f}]")
    print(f"  Y: [{points3D_all[:,1].min():.2f}, {points3D_all[:,1].max():.2f}]")
    print(f"  Z: [{points3D_all[:,2].min():.2f}, {points3D_all[:,2].max():.2f}]")
    
    # Statistical outlier removal
    mean = np.mean(points3D_all, axis=0)
    std = np.std(points3D_all, axis=0)
    dist_from_mean = np.abs(points3D_all - mean)
    outlier_mask = np.all(dist_from_mean < 2.5 * std, axis=1)  # More lenient
    
    points3D_filtered = points3D_all[outlier_mask]
    colors_filtered = colors_all[outlier_mask]
    
    print(f"Points after outlier removal: {len(points3D_filtered)}/{len(points3D_all)}")
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points
    ax.scatter(points3D_filtered[:, 0], 
               points3D_filtered[:, 1], 
               points3D_filtered[:, 2],
               c=colors_filtered, s=2, alpha=0.6)
    
    # Plot camera positions
    cam_positions = []
    for idx, (R, t) in enumerate(camera_poses):
        cam_center = (-R.T @ t).flatten()
        cam_center = np.array([cam_center[0], cam_center[1], cam_center[2]])
        cam_positions.append(cam_center)
        x, y, z = float(cam_center[0]), float(cam_center[1]), float(cam_center[2])
        ax.scatter(x, y, z, c='red', marker='o', s=150, edgecolors='black', linewidths=2)
        ax.text(x, y, z, f'  C{idx}', fontsize=9, weight='bold')
    
    # Camera trajectory
    if len(cam_positions) > 1:
        cam_positions = np.vstack(cam_positions)
        ax.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], 
                'r-', linewidth=3, label='Camera trajectory', alpha=0.7)
    
    ax.set_xlabel('X', fontsize=14, weight='bold')
    ax.set_ylabel('Y', fontsize=14, weight='bold')
    ax.set_zlabel('Z', fontsize=14, weight='bold')
    ax.set_title(f'3D Reconstruction: {len(points3D_filtered)} points, {len(camera_poses)} cameras', 
                 fontsize=16, weight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
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
    plt.savefig('reconstruction_3d_improved.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n3D reconstruction saved to 'reconstruction_3d_improved.png'")
    
    # Save PLY
    with open('point_cloud_improved.ply', 'w') as f:
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
    
    print(f"Point cloud saved to 'point_cloud_improved.ply'")
    
    # Print recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS FOR BETTER RECONSTRUCTION:")
    print(f"{'='*70}")
    if len(points3D_filtered) < 500:
        print("⚠ VERY FEW POINTS - Your reconstruction needs improvement!")
        print("\n✓ CAPTURE TECHNIQUE:")
        print("  - Take 30-50 images (you have only 14)")
        print("  - Ensure 70-80% overlap between consecutive images")
        print("  - Move in a circular pattern around the object")
        print("  - Keep camera ~1-2 meters from object")
        print("\n✓ SCENE SETUP:")
        print("  - Use well-lit, textured objects")
        print("  - Avoid reflective/transparent surfaces")
        print("  - Add checkerboard/markers to plain surfaces")
        print("  - Keep background static and textured")
        print("\n✓ CAMERA SETTINGS:")
        print("  - Use fixed focal length (no zoom)")
        print("  - Keep consistent exposure/focus")
        print("  - Avoid motion blur")
    
    print(f"\n{'='*70}")
    print("RECONSTRUCTION COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
