import cv2
import numpy as np

def load_image(image_path):
    return cv2.imread(image_path)

def undistort_image(image, camera_matrix, dist_coeffs):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    return undistorted_img[y:y+h, x:x+w]

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return edges

def find_reference_object(edges, known_dimension):
    p1, p2 = (100, 100), (200, 200)  # Example coordinates of reference object
    pixel_distance = np.linalg.norm(np.array(p1) - np.array(p2))
    scale_factor = known_dimension / pixel_distance
    return scale_factor

def measure_object(edges, scale_factor):
    p1, p2 = (300, 300), (400, 400)  # Example of a measurement to be made
    pixel_distance = np.linalg.norm(np.array(p1) - np.array(p2))
    real_world_measurement = pixel_distance * scale_factor
    return real_world_measurement

def draw_detected_edges(image, edges):
    # Convert edges to color so we can draw them in red
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Set red color to the edges
    edges_colored[edges != 0] = [0, 0, 255]  # Red color (BGR format)
    return cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

def highlight_key_lines(image, edges):
    # Use Hough Transform to detect straight lines (e.g., walls)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw lines in red (BGR)
    
    return image

def main(image_path, known_dimension, camera_matrix, dist_coeffs):
    image = load_image(image_path)
    undistorted_image = undistort_image(image, camera_matrix, dist_coeffs)
    edges = detect_edges(undistorted_image)

    # Find the scale factor based on a known reference dimension (e.g., height of a door)
    scale_factor = find_reference_object(edges, known_dimension)
    
    # Measure another object in the image
    measurement = measure_object(edges, scale_factor)
    print(f'Approximate real-world measurement: {measurement} meters')

    # Highlight detected edges in red
    highlighted_image = draw_detected_edges(undistorted_image.copy(), edges)
    
    # Highlight key lines like room corners and walls in red
    highlighted_image = highlight_key_lines(highlighted_image, edges)
    
    # Save the image with the highlighted edges and lines
    output_image_path = "highlighted_edges.jpg"
    cv2.imwrite(output_image_path, highlighted_image)
    print(f"Image with highlighted edges saved to {output_image_path}")

if __name__ == "__main__":
    image_path = "test.jpg"
    known_dimension = 2.0  # Reference object dimension, e.g., 2 meters for a door
    camera_matrix = np.array([[1.2e+03, 0, 640], [0, 1.2e+03, 360], [0, 0, 1]])
    dist_coeffs = np.array([-0.2, 0.1, 0, 0, 0])

    main(image_path, known_dimension, camera_matrix, dist_coeffs)