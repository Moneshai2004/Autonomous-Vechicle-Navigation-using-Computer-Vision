import cv2
import numpy as np

# Path to the video file
input_path = "vision-data/challenge.mp4"
output_path = "output_video_lane.mp4"  # Path for the output video

# Load video
cap = cv2.VideoCapture(input_path)

# Check if video is loaded
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for output
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Grayscale conversion
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Gaussian blur
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Canny edge detection
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# Define Region of Interest
def get_aoi(img):
    rows, cols = img.shape[:2]
    mask = np.zeros_like(img)
    left_bottom = [cols * 0.1, rows]
    right_bottom = [cols * 0.95, rows]
    left_top = [cols * 0.4, rows * 0.6]
    right_top = [cols * 0.6, rows * 0.6]
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

# Hough Transformation Lines Detection
def get_hough_lines(img, rho=1, theta=np.pi/180, threshold=20, min_line_len=20, max_line_gap=300):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# Function to draw lines on the image
def draw_lines(image, lines, color=(0, 255, 0), thickness=5):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

# Process each frame to detect lanes
def process_image(frame):
    gray_img = grayscale(frame)
    blur_img = gaussian_blur(gray_img, 5)
    edges = canny(blur_img, 50, 150)
    aoi_img = get_aoi(edges)
    hough_lines = get_hough_lines(aoi_img)
    
    # Draw detected lane lines
    lane_frame = draw_lines(frame, hough_lines)

    # Movement Detection
    movement_state = "Stop"
    if hough_lines is not None:
        # Analyze lines to determine direction
        left_lines = [line for line in hough_lines if line[0][0] < frame.shape[1] / 2]
        right_lines = [line for line in hough_lines if line[0][0] >= frame.shape[1] / 2]

        if left_lines and not right_lines:
            movement_state = "Left"
        elif right_lines and not left_lines:
            movement_state = "Right"
        elif left_lines and right_lines:
            movement_state = "Straight"
        else:
            movement_state = "Stop"

    # Put movement state on frame
    cv2.putText(lane_frame, f'Movement: {movement_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return lane_frame

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_image(frame)
    
    # Display the processed frame
    cv2.imshow("Final Result - Lane Detection", processed_frame)

    # Wait to maintain the original frame rate
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

    out.write(processed_frame)

# Release resources
cap.release()
out.release()

print("Video processing complete. Output saved as:", output_path)

cv2.destroyAllWindows()
