import cv2
import numpy as np

# Check OpenCV version
opencv_ver = cv2.__version__.split('.')[0]
print(f"Current version of OpenCV is {opencv_ver}")

# Path to the video file
input_path = "vision-data/M6 Motorway Traffic.mp4"
output_path = "output_video.mp4"  # Path for the output video

# Load video
cap = cv2.VideoCapture(input_path)

# Create background subtractor
bg = cv2.createBackgroundSubtractorMOG2()

# Get video properties for output
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    bgmask = bg.apply(frame)

    # Find contours of the moving objects
    contours, hierarchy = cv2.findContours(bgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each contour, get bounding box, and draw it
    for c in contours:
        if cv2.contourArea(c) < 800:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the background mask and final result
    cv2.imshow("Background Mask", bgmask)
    cv2.imshow("Final Result", frame)

    # Wait to maintain the original frame rate
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()

# Check if frame was processed before saving images
if 'frame' in locals() and frame is not None and frame.size > 0:
    # Save the final background mask and frame if needed
    cv2.imwrite('background_mask.png', bgmask)
    cv2.imwrite('final_result.png', frame)

print("Video processing complete. Output saved as:", output_path)

cv2.destroyAllWindows()
