import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def setup_matplotlib_for_plotting():
    """Setup matplotlib for plotting."""
    import warnings
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", 
                                         "PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

def clean_point_cloud(points3d, colors=None, remove_outliers=True, 
                     depth_range=[0.5, 10.0], keep_percentage=0.8):
    """
    Clean point cloud to focus on main object
    
    Args:
        points3d: Nx3 array of 3D points
        colors: Nx3 array of colors (optional)
        remove_outliers: Remove statistical outliers
        depth_range: [min_depth, max_depth] from camera origin
        keep_percentage: Keep only this percentage of points closest to center
    """
    
    print("🔧 Cleaning point cloud...")
    print(f"   Original points: {len(points3d)}")
    
    # 1. Statistical outlier removal (3-sigma rule)
    if remove_outliers:
        mean = np.mean(points3d, axis=0)
        std = np.std(points3d, axis=0)
        dist_from_mean = np.abs(points3d - mean)
        outlier_mask = np.all(dist_from_mean < 2.5 * std, axis=1)
        
        points3d = points3d[outlier_mask]
        if colors is not None:
            colors = colors[outlier_mask]
        
        print(f"   After outlier removal: {len(points3d)}")
    
    # 2. Depth filtering (remove points too far from camera)
    # Assume camera at origin (0,0,0) or use median position
    if depth_range:
        distances = np.linalg.norm(points3d, axis=1)
        depth_mask = (distances >= depth_range[0]) & (distances <= depth_range[1])
        points3d = points3d[depth_mask]
        if colors is not None:
            colors = colors[depth_mask]
        print(f"   After depth filtering: {len(points3d)}")
    
    # 3. Central focus - keep only points closest to center
    center = np.median(points3d, axis=0)
    distances_to_center = np.linalg.norm(points3d - center, axis=1)
    
    # Keep only the closest X% of points
    keep_count = int(len(points3d) * keep_percentage)
    sort_indices = np.argsort(distances_to_center)
    central_mask = sort_indices[:keep_count]
    
    points3d_clean = points3d[central_mask]
    if colors is not None:
        colors_clean = colors[central_mask]
        print(f"   After central focus: {len(points3d_clean)}")
        return points3d_clean, colors_clean
    
    return points3d_clean, None

def create_depth_histogram(points3d):
    """Show depth distribution to help set depth range"""
    distances = np.linalg.norm(points3d, axis=1)
    
    setup_matplotlib_for_plotting()
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(distances, bins=100, alpha=0.7, edgecolor='black')
    plt.xlabel('Distance from origin')
    plt.ylabel('Number of points')
    plt.title('Point Cloud Depth Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(distances)
    plt.ylabel('Distance from origin')
    plt.title('Depth Statistics')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('depth_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Distance statistics:")
    print(f"  Min: {distances.min():.2f}")
    print(f"  Max: {distances.max():.2f}")
    print(f"  Median: {np.median(distances):.2f}")
    print(f"  25th percentile: {np.percentile(distances, 25):.2f}")
    print(f"  75th percentile: {np.percentile(distances, 75):.2f}")

def visualize_cleaning_process(points3d, colors=None, cleaned_points=None, cleaned_colors=None):
    """Show before/after cleaning"""
    
    setup_matplotlib_for_plotting()
    fig = plt.figure(figsize=(20, 8))
    
    # Original point cloud
    ax1 = fig.add_subplot(131, projection='3d')
    if colors is not None:
        ax1.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], 
                   c=colors, s=0.5, alpha=0.3)
    else:
        ax1.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], s=0.5, alpha=0.3)
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Depth histogram
    distances = np.linalg.norm(points3d, axis=1)
    ax2 = fig.add_subplot(132)
    ax2.hist(distances, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(np.median(distances), color='red', linestyle='--', label=f'Median: {np.median(distances):.1f}')
    ax2.axvline(np.percentile(distances, 25), color='orange', linestyle='--', label=f'25th %ile: {np.percentile(distances, 25):.1f}')
    ax2.axvline(np.percentile(distances, 75), color='orange', linestyle='--', label=f'75th %ile: {np.percentile(distances, 75):.1f}')
    ax2.set_xlabel('Distance from origin')
    ax2.set_ylabel('Point count')
    ax2.set_title('Depth Distribution Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cleaned point cloud (if provided)
    if cleaned_points is not None:
        ax3 = fig.add_subplot(133, projection='3d')
        if cleaned_colors is not None:
            ax3.scatter(cleaned_points[:, 0], cleaned_points[:, 1], cleaned_points[:, 2],
                       c=cleaned_colors, s=2, alpha=0.8)
        else:
            ax3.scatter(cleaned_points[:, 0], cleaned_points[:, 1], cleaned_points[:, 2], s=2, alpha=0.8)
        ax3.set_title(f'Cleaned Point Cloud ({len(cleaned_points)} points)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
    else:
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.text(0.5, 0.5, 0.5, 'Run analysis first\nto see depth statistics', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=14)
        ax3.set_title('Cleaned Point Cloud')
    
    plt.tight_layout()
    plt.savefig('point_cloud_cleaning_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()

def main():
    print("🧹 POINT CLOUD CLEANING TOOL")
    print("="*50)
    
    # Load your point cloud
    print("Loading point cloud...")
    try:
        # Try loading from PLY file
        points = []
        colors = []
        with open('point_cloud_improved.ply', 'r') as f:
            lines = f.readlines()
            header_end = 0
            for i, line in enumerate(lines):
                if line.strip() == 'end_header':
                    header_end = i + 1
                    break
            
            for line in lines[header_end:]:
                data = line.strip().split()
                if len(data) >= 6:
                    x, y, z, r, g, b = map(float, data[:6])
                    points.append([x, y, z])
                    colors.append([r/255.0, g/255.0, b/255.0])
        
        points = np.array(points)
        colors = np.array(colors)
        print(f"✅ Loaded {len(points)} points from PLY file")
        
    except FileNotFoundError:
        try:
            # Alternative: load from numpy array
            data = np.load('points3D_all.npy')
            points = data
            colors = None
            print(f"✅ Loaded {len(points)} points from NPY file")
        except FileNotFoundError:
            print("❌ Could not find point cloud file!")
            print("   Expected: 'point_cloud_improved.ply' or 'points3D_all.npy'")
            return
    
    # Analyze depth distribution
    create_depth_histogram(points)
    
    # Show original point cloud
    visualize_cleaning_process(points, colors)
    
    # Interactive cleaning parameters (you can adjust these)
    print(f"\n📊 Point Cloud Analysis:")
    distances = np.linalg.norm(points, axis=1)
    print(f"   Distance range: {distances.min():.2f} - {distances.max():.2f}")
    print(f"   Median distance: {np.median(distances):.2f}")
    
    # Suggest depth range based on your shooting distance (~3 feet = ~0.9 meters)
    suggested_min = np.percentile(distances, 10)  # Remove very close background
    suggested_max = np.percentile(distances, 90)  # Remove very far background
    
    print(f"\n🎯 Recommended depth range: {suggested_min:.2f} - {suggested_max:.2f}")
    
    # Clean the point cloud
    cleaned_points, cleaned_colors = clean_point_cloud(
        points, colors,
        depth_range=[suggested_min, suggested_max],
        keep_percentage=0.6  # Keep central 60% of points
    )
    
    # Show cleaned result
    visualize_cleaning_process(points, colors, cleaned_points, cleaned_colors)
    
    # Save cleaned point cloud
    print(f"\n💾 Saving cleaned point cloud...")
    
    with open('point_cloud_cleaned.ply', 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(cleaned_points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        
        for i, point in enumerate(cleaned_points):
            if cleaned_colors is not None:
                r, g, b = (cleaned_colors[i] * 255).astype(int)
            else:
                r = g = b = 128  # Gray color
            f.write(f'{point[0]} {point[1]} {point[2]} {r} {g} {b}\n')
    
    print(f"✅ Saved cleaned point cloud: 'point_cloud_cleaned.ply'")
    print(f"   Original: {len(points)} points")
    print(f"   Cleaned: {len(cleaned_points)} points ({100*len(cleaned_points)/len(points):.1f}%)")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. View 'depth_analysis.png' to see point distribution")
    print(f"   2. View 'point_cloud_cleaning_analysis.png' for visual comparison")
    print(f"   3. Load 'point_cloud_cleaned.ply' in MeshLab for 3D visualization")
    print(f"   4. Adjust depth_range parameter if object is still not clear")

if __name__ == '__main__':
    main()
