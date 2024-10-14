import cv2
import numpy as np

# image_path = "test.jpg"

def load_image(image_path):
    # Load image using OpenCV
    return cv2.imread(image_path)

def undistort_image(image, camera_matrix, dist_coeffs):
    # Correct lens distortion using the provided camera calibration parameters
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    return undistorted_img[y:y+h, x:x+w]  # Crop to ROI

def detect_edges(image):
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return edges

def find_reference_object(edges, known_dimension):
    # Detect the reference object, e.g., using HoughLines or contours (assuming user input for now)
    # Placeholder for user-specified points of the reference object
    p1, p2 = (100, 100), (200, 200)  # Example coordinates of the reference object in the image
    pixel_distance = np.linalg.norm(np.array(p1) - np.array(p2))
    scale_factor = known_dimension / pixel_distance
    return scale_factor

def measure_object(edges, scale_factor):
    # Detect other objects and measure their dimensions in pixels, then scale them
    p1, p2 = (300, 300), (400, 400)  # Example of a measurement to be made in the image
    pixel_distance = np.linalg.norm(np.array(p1) - np.array(p2))
    real_world_measurement = pixel_distance * scale_factor
    return real_world_measurement

def main(image_path, known_dimension, camera_matrix, dist_coeffs):
    image = load_image(image_path)
    undistorted_image = undistort_image(image, camera_matrix, dist_coeffs)
    edges = detect_edges(undistorted_image)

    # Assume we know a reference object's dimension, e.g., a door height of 2 meters
    scale_factor = find_reference_object(edges, known_dimension)

    # Measure another object in the image
    measurement = measure_object(edges, scale_factor)
    
    print(f'Approximate real-world measurement: {measurement} meters')

if __name__ == "__main__":
    # Example inputs
    image_path = "test.jpg"
    known_dimension = 2.0  # Reference object dimension, e.g., 2 meters for a door
    # Example camera calibration data (would need to be pre-calculated)
    camera_matrix = np.array([[1.2e+03, 0, 640], [0, 1.2e+03, 360], [0, 0, 1]])
    dist_coeffs = np.array([-0.2, 0.1, 0, 0, 0])

    main(image_path, known_dimension, camera_matrix, dist_coeffs)
