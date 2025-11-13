import cv2
import numpy as np
import glob
import os

# Configuration
ARUCO_DICT = cv2.aruco.DICT_4X4_1000
SQUARES_X = 11  # Number of squares in X direction
SQUARES_Y = 16  # Number of squares in Y direction
SQUARE_LENGTH = 0.021  # Total square size (21mm)
MARKER_LENGTH = 0.015  # Marker size (15mm)
PATH_TO_IMAGES = "./photo_calib/"

def calibrate_camera():
    """
    Perform camera calibration using ChArUco board images.
    
    Returns:
        camera_matrix: Intrinsic camera matrix
        dist_coeffs: Distortion coefficients
        rvecs: Rotation vectors for each image
        tvecs: Translation vectors for each image
    """
    
    # Initialize ArUco dictionary and ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        aruco_dict, 
        ids=np.arange(30, 118)
    )
    board.setLegacyPattern(True) 
    image = cv2.aruco.CharucoBoard.generateImage(board, outSize=(1920,  1080))
    cv2.imshow("ChArUco Board", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Board configuration: {SQUARES_X}x{SQUARES_Y} squares")
    print(f"Expected markers: {(SQUARES_X//2) * (SQUARES_Y//2) + ((SQUARES_X+1)//2) * ((SQUARES_Y+1)//2)}")
    print(f"Square length: {SQUARE_LENGTH}m, Marker length: {MARKER_LENGTH}m")
    
    # Initialize ArUco detector parameters
    parameters = cv2.aruco.DetectorParameters()
    # Adjust parameters for better detection
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.03
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 3
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Prepare lists to store object points and image points
    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None
    
    # Get all images from the calibration folder
    image_files = glob.glob(os.path.join(PATH_TO_IMAGES, "*.jpg")) + \
                  glob.glob(os.path.join(PATH_TO_IMAGES, "*.png")) + \
                    glob.glob(os.path.join(PATH_TO_IMAGES, "*.JPEG")) 
    if len(image_files) == 0:
        print(f"No images found in {PATH_TO_IMAGES}")
        return None, None, None, None
    
    print(f"Found {len(image_files)} images for calibration")
    
    # Process each image
    for image_file in image_files:
        img = cv2.imread(image_file)
        
        # Resize image for better detection (scale down large images)
        max_dimension = 1920
        height, width = img.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            print(f"Resized {os.path.basename(image_file)} from {width}x{height} to {new_width}x{new_height}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = gray.shape[::-1]
        
        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            print(f"  Detected {len(ids)} ArUco markers")
            
            # Try to interpolate ChArUco corners
            try:
                retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board
                )
                
                if retval > 0:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                    print(f"✓ {os.path.basename(image_file)}: Detected {retval} ChArUco corners")
                else:
                    print(f"✗ {os.path.basename(image_file)}: interpolateCornersCharuco returned {retval} corners")
            except Exception as e:
                print(f"✗ {os.path.basename(image_file)}: Error during interpolation - {e}")
        else:
            print(f"✗ {os.path.basename(image_file)}: No ArUco markers detected")
    
    if len(all_charuco_corners) == 0:
        print("No valid ChArUco corners detected in any image!")
        return None, None, None, None
    
    print(f"\nCalibrating camera using {len(all_charuco_corners)} images...")
    
    # Calibrate camera using ChArUco
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        image_size,
        None,
        None
    )
    
    if ret:
        print("\n" + "="*50)
        print("CALIBRATION SUCCESSFUL")
        print("="*50)
        print(f"\nReprojection Error: {ret:.4f} pixels")
        print("\nCamera Matrix (K):")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        print("="*50)
        
        # Save calibration results
        np.savez(
            'calibration.npz',
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rvecs=rvecs,
            tvecs=tvecs,
            reprojection_error=ret
        )
        print("\nCalibration saved to 'calibration.npz'")
        
        return camera_matrix, dist_coeffs, rvecs, tvecs
    else:
        print("\nCalibration failed!")
        return None, None, None, None


def load_calibration(filename='calibration.npz'):
    """
    Load calibration data from file.
    
    Args:
        filename: Path to calibration file
        
    Returns:
        camera_matrix, dist_coeffs
    """
    with np.load(filename) as data:
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        print("Calibration loaded successfully")
        print(f"Reprojection Error: {data['reprojection_error']:.4f} pixels")
        return camera_matrix, dist_coeffs


if __name__ == "__main__":
    # Perform calibration
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera()
    
    if camera_matrix is not None:
        print("\nCalibration complete! Use load_calibration() to load the results.")