import numpy as np
import cv2

def optimized_feature_matching(img1, img2, K, dist_coeffs):
    """
    Optimized feature matching for better reconstruction
    """
    # Enhanced SIFT with better parameters
    sift = cv2.SIFT_create(
        nfeatures=8000,        # More features
        contrastThreshold=0.02, # Lower = more features
        edgeThreshold=10,       # Higher = more features
        sigma=1.6
    )
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # FLANN matcher with tuned parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=100, eps=0, sorted=True)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # KNN matching with Lowe ratio test
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:  # More lenient
            good_matches.append(m)
    
    # Additional geometric filtering
    if len(good_matches) > 50:
        pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)
        
        # Find fundamental matrix and filter outliers
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
        if mask is not None:
            inliers = mask.ravel() == 1
            good_matches = [m for i, m in enumerate(good_matches) if inliers[i]]
    
    return kp1, kp2, good_matches

def get_optimal_depth_filter(distances, shooting_distance=0.9):
    """
    Calculate optimal depth range for filtering
    shooting_distance: your camera-to-object distance in meters
    """
    # Remove extreme outliers
    q25, q75 = np.percentile(distances, [10, 90])
    
    # Focus on main object depth (around your shooting distance)
    obj_depth_min = max(0.5, q25)
    obj_depth_max = min(shooting_distance * 2.0, q75)
    
    return obj_depth_min, obj_depth_max

def print_capture_recommendations():
    """Print specific recommendations for better reconstruction"""
    
    print("📸 OPTIMIZED CAPTURE RECOMMENDATIONS")
    print("=" * 60)
    print(f"📐 Your setup:")
    print(f"   - Object size: ~2ft × 1.5ft")
    print(f"   - Camera distance: ~3ft (0.9m)")
    print(f"   - Current points: 26,000+ (good!)")
    print()
    
    print(f"🎯 CRITICAL ISSUES & SOLUTIONS:")
    print(f"   1. BACKGROUND CLUTTER")
    print(f"      Problem: Floor, walls, table being reconstructed")
    print(f"      ✅ Solution: Use solid color background (white/gray)")
    print(f"      ✅ Solution: Shoot on plain surface (mat/tablecloth)")
    print()
    
    print(f"   2. SCENE COMPLEXITY")
    print(f"      Problem: Mixed calibration board + object")
    print(f"      ✅ Solution: Separate into 2 datasets:")
    print(f"         - calibration_images/ (just board, ~20 images)")
    print(f"         - object_images/ (just object, ~75 images)")
    print()
    
    print(f"   3. CAMERA MOVEMENT PATTERN")
    print(f"      ✅ Optimal for 3ft distance:")
    print(f"         • Start 45° to object")
    print(f"         • Move clockwise in circle (20-30 steps)")
    print(f"         • Each step ~15° rotation + slight position change")
    print(f"         • Keep constant distance (~3ft)")
    print(f"         • Add 3-5 shots from different heights")
    print()
    
    print(f"   4. OVERLAP & COVERAGE")
    print(f"      ✅ Requirements:")
    print(f"         • 80-85% overlap between consecutive shots")
    print(f"         • 360° coverage of object")
    print(f"         • No major gaps in coverage")
    print(f"         • Consistent lighting throughout")
    print()
    
    print(f"   5. OBJECT PREPARATION")
    print(f"      ✅ For best results:")
    print(f"         • Ensure good texture on all surfaces")
    print(f"         • Add markers/tape if surfaces are too plain")
    print(f"         • Use consistent, diffused lighting")
    print(f"         • Keep object stationary during capture")
    print()
    
    print(f"🔧 POST-PROCESSING RECOMMENDATIONS:")
    print(f"   • Use depth filtering to remove background")
    print(f"   • Statistical outlier removal")
    print(f"   • Manual cleanup in MeshLab/CloudCompare")
    print(f"   • Mesh generation for solid surfaces")

def print_example_workflow():
    """Show example workflow for better results"""
    
    print(f"\n📋 EXAMPLE WORKFLOW FOR BETTER RECONSTRUCTION:")
    print(f"=" * 60)
    print(f"1. SETUP (10 minutes)")
    print(f"   □ Place object on plain background")
    print(f"   □ Set up consistent lighting")
    print(f"   □ Set camera to fixed settings")
    print()
    print(f"2. CALIBRATION (5 minutes)")
    print(f"   □ Take 20 images of just the calibration board")
    print(f"   □ Use these for camera calibration")
    print(f"   □ Store in 'calibration/' folder")
    print()
    print(f"3. OBJECT CAPTURE (10 minutes)")
    print(f"   □ Position camera 3ft from object")
    print(f"   □ Take 75+ images moving in circular pattern")
    print(f"   □ Ensure good overlap between shots")
    print(f"   □ Store in 'object_reconstruction/' folder")
    print()
    print(f"4. PROCESSING (5 minutes)")
    print(f"   □ Run reconstruction on object images only")
    print(f"   □ Apply depth filtering")
    print(f"   □ Clean point cloud")
    print(f"   □ Generate final 3D model")
    print()
    print(f"5. EVALUATION")
    print(f"   □ Check point cloud density (>5000 points ideal)")
    print(f"   □ Verify camera trajectory is smooth")
    print(f"   □ Look for object vs background separation")

if __name__ == '__main__':
    print_capture_recommendations()
    print_example_workflow()
    
    print(f"\n📊 DEPTH ANALYSIS FOR YOUR DATA:")
    print(f"   Since you're shooting from ~3ft (0.9m):")
    print(f"   • Main object should be at 0.8-1.2m depth")
    print(f"   • Background (floor/walls) at 2.0-5.0m+")
    print(f"   • Use depth filter to remove <0.7m and >2.0m points")
    print(f"   • This should isolate your main object!")
