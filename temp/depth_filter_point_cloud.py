#!/usr/bin/env python3
"""
Depth-based point cloud filtering script for 3D reconstruction cleanup.
Filters points based on distance from camera origin to isolate main objects
from background clutter (floor, walls, table surfaces).

Usage:
    python depth_filter_point_cloud.py input.ply --depth_min 0.8 --depth_max 1.2 --output cleaned.ply
    python depth_filter_point_cloud.py input.npy --depth_min 0.8 --depth_max 1.2 --output cleaned.npy
"""

import numpy as np
import argparse
import os
import sys
from pathlib import Path

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. PLY saving will be limited.")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Warning: Matplotlib not available. Visualizations disabled.")


def load_point_cloud(file_path):
    """Load point cloud from PLY or NPY file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading point cloud from: {file_path}")
    
    if file_path.suffix.lower() == '.ply':
        if OPEN3D_AVAILABLE:
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            return points, colors
        else:
            raise ImportError("Open3D is required to load PLY files")
    
    elif file_path.suffix.lower() == '.npy':
        data = np.load(file_path)
        # Handle different NPY formats
        if data.ndim == 2 and data.shape[1] >= 3:
            if data.shape[1] == 3:
                # Just XYZ coordinates
                points = data
                colors = None
            elif data.shape[1] == 6:
                # XYZ + RGB
                points = data[:, :3]
                colors = data[:, 3:6]
            elif data.shape[1] == 4:
                # XYZ + weight/homogeneous coordinate
                points = data[:, :3]
                colors = None
            else:
                points = data[:, :3]
                colors = None
        elif data.ndim == 1 and len(data) > 0:
            # Might be structured array or flattened
            if hasattr(data.item(0), '__len__'):
                data_item = data.item(0)
                if len(data_item) >= 3:
                    points = np.array([np.array(item) for item in data])
                    colors = None
                else:
                    raise ValueError(f"Unexpected data format in NPY file: {data.shape}")
            else:
                raise ValueError(f"Unexpected NPY format: {data.shape}")
        else:
            raise ValueError(f"Unexpected NPY format: {data.shape}")
        
        return points, colors
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def filter_by_depth(points, depth_min=0.8, depth_max=1.2, return_indices=False):
    """Filter points based on distance from camera origin (0,0,0)."""
    if points.size == 0:
        if return_indices:
            return np.array([]), np.array([], dtype=bool), np.array([], dtype=int)
        else:
            return np.array([]), np.array([], dtype=bool)
    
    # Calculate depth (Euclidean distance from origin)
    depths = np.linalg.norm(points, axis=1)
    
    # Create boolean mask for points within depth range
    mask = (depths >= depth_min) & (depths <= depth_max)
    
    # Filter points
    filtered_points = points[mask]
    filtered_depths = depths[mask]
    
    print(f"Depth filtering results:")
    print(f"  Original points: {len(points):,}")
    print(f"  Points within {depth_min:.1f}-{depth_max:.1f}m: {len(filtered_points):,}")
    print(f"  Filtered out: {len(points) - len(filtered_points):,} ({(len(points) - len(filtered_points))/len(points)*100:.1f}%)")
    
    if len(filtered_points) > 0:
        print(f"  Depth range of kept points: {filtered_depths.min():.3f} - {filtered_depths.max():.3f}m")
        print(f"  Mean depth: {filtered_depths.mean():.3f}m")
    
    if return_indices:
        return filtered_points, mask, np.where(mask)[0]
    else:
        return filtered_points, mask


def save_point_cloud(points, colors, output_path, file_format=None):
    """Save point cloud to PLY or NPY file."""
    output_path = Path(output_path)
    
    if file_format is None:
        file_format = output_path.suffix.lower()
    
    print(f"Saving filtered point cloud to: {output_path}")
    
    if file_format == '.ply':
        if OPEN3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None and len(colors) == len(points):
                pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(str(output_path), pcd)
        else:
            # Fallback: save as PLY with just points
            save_ply_simple(output_path, points)
    
    elif file_format == '.npy':
        if colors is not None and len(colors) == len(points):
            # Save with colors
            data = np.column_stack([points, colors])
        else:
            # Save just points
            data = points
        np.save(output_path, data)
    
    else:
        raise ValueError(f"Unsupported output format: {file_format}")


def save_ply_simple(output_path, points):
    """Simple PLY writer that only saves point coordinates."""
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def create_visualization(points, filtered_points, output_dir="."):
    """Create depth distribution visualization."""
    if not PLOT_AVAILABLE or len(points) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Depth histogram
    depths_all = np.linalg.norm(points, axis=1)
    depths_filtered = np.linalg.norm(filtered_points, axis=1) if len(filtered_points) > 0 else np.array([])
    
    ax1.hist(depths_all, bins=50, alpha=0.7, label='All points', color='lightblue', edgecolor='black')
    if len(depths_filtered) > 0:
        ax1.hist(depths_filtered, bins=50, alpha=0.7, label='Filtered points', color='red', edgecolor='black')
    
    ax1.set_xlabel('Depth (meters)')
    ax1.set_ylabel('Number of points')
    ax1.set_title('Depth Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3D scatter of filtered points
    if len(filtered_points) > 0:
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], 
                   c=depths_filtered, cmap='viridis', s=1, alpha=0.6)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title('Filtered Point Cloud (colored by depth)')
        
        # Set equal aspect ratio
        max_range = np.array([filtered_points[:,0].max()-filtered_points[:,0].min(),
                             filtered_points[:,1].max()-filtered_points[:,1].min(),
                             filtered_points[:,2].max()-filtered_points[:,2].min()]).max() / 2.0
        mid_x = (filtered_points[:,0].max()+filtered_points[:,0].min()) * 0.5
        mid_y = (filtered_points[:,1].max()+filtered_points[:,1].min()) * 0.5
        mid_z = (filtered_points[:,2].max()+filtered_points[:,2].min()) * 0.5
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        ax2.text(0.5, 0.5, 'No points in filtered range', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Filtered Point Cloud')
    
    plt.tight_layout()
    vis_path = os.path.join(output_dir, "depth_filtering_visualization.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {vis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter point cloud by depth range to isolate main objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter PLY file with default 0.8-1.2m range
  python depth_filter_point_cloud.py point_cloud.ply
  
  # Filter with custom range and save as NPY
  python depth_filter_point_cloud.py points3D_all.npy --depth_min 0.7 --depth_max 1.3 --output cleaned.npy
  
  # Get detailed statistics only
  python depth_filter_point_cloud.py point_cloud.ply --stats_only
        """
    )
    
    parser.add_argument('input', help='Input point cloud file (.ply or .npy)')
    parser.add_argument('--depth_min', type=float, default=0.8, 
                       help='Minimum depth in meters (default: 0.8)')
    parser.add_argument('--depth_max', type=float, default=1.2, 
                       help='Maximum depth in meters (default: 1.2)')
    parser.add_argument('--output', '-o', help='Output file path (default: input_filtered.ply)')
    parser.add_argument('--stats_only', action='store_true', 
                       help='Only show statistics, do not filter or save')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create depth distribution visualization')
    
    args = parser.parse_args()
    
    # Validate depth range
    if args.depth_min >= args.depth_max:
        print("Error: depth_min must be less than depth_max")
        sys.exit(1)
    
    # Set default output filename
    if args.output is None:
        input_path = Path(args.input)
        args.output = f"{input_path.stem}_filtered{input_path.suffix}"
    
    try:
        # Load point cloud
        points, colors = load_point_cloud(args.input)
        
        if len(points) == 0:
            print("Error: No points found in input file")
            sys.exit(1)
        
        print(f"Loaded {len(points):,} points")
        
        # Filter by depth
        filtered_points, mask, indices = filter_by_depth(
            points, args.depth_min, args.depth_max, return_indices=True
        )
        
        if args.stats_only:
            return
        
        # Filter colors if available
        filtered_colors = None
        if colors is not None and len(colors) == len(points):
            filtered_colors = colors[mask]
            print(f"Preserved colors for {len(filtered_colors):,} points")
        
        if len(filtered_points) == 0:
            print("Warning: No points remain after filtering!")
            print("Consider adjusting depth_min and depth_max parameters")
            return
        
        # Save filtered point cloud
        save_point_cloud(filtered_points, filtered_colors, args.output)
        
        # Create visualization if requested
        if args.visualize:
            create_visualization(points, filtered_points, output_dir=".")
        
        print(f"\nFiltering complete!")
        print(f"Output saved to: {args.output}")
        print(f"Original: {len(points):,} points")
        print(f"Filtered: {len(filtered_points):,} points")
        print(f"Reduction: {(1 - len(filtered_points)/len(points))*100:.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()