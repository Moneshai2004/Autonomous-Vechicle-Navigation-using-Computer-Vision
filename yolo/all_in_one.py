import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("/home/moni/Desktop/automous vechicle/yolo/yolov3.weights", 
                       "/home/moni/Desktop/automous vechicle/yolo/yolov3.cfg")

# Load class names
with open("/home/moni/Desktop/automous vechicle/yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get layer names and output layer indices
layer_names = net.getLayerNames()
out_layers_indices = net.getUnconnectedOutLayers()
if isinstance(out_layers_indices, np.ndarray):
    output_layers = [layer_names[i - 1] for i in out_layers_indices.flatten()]
else:
    output_layers = [layer_names[i - 1] for i in out_layers_indices]

# Load video
input_path = "/home/moni/Desktop/automous vechicle/vision-data/samplevideo.webm"  # Update this path to your video
cap = cv2.VideoCapture(input_path)

# Define movement instructions (You can change these dynamically later)
movement_instruction = "Accelarate"

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object Detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Detect objects
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(detection[0] * frame.shape[1]), int(detection[1] * frame.shape[0])
                w, h = int(detection[2] * frame.shape[1]), int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Resize the frame for display
    frame_resized = cv2.resize(frame, (900, 600))

    cv2.putText(frame_resized, f'Movement: {movement_instruction}', (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display processed frame
    cv2.imshow("Frame", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
