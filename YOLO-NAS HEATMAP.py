import numpy as np
import torch
import cvzone
from super_gradients.training import models
import cv2
import math

# Open the video file
cap = cv2.VideoCapture('birds.mp4')

# Check if GPU (CUDA) is available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Load the YOLO model with NAS architecture and COCO pretrained weights
model = models.get('yolo_nas_s', pretrained_weights='coco').to(device)

# Initialize heatmap dimensions
h, w = 500, 700
globalimagearray = np.zeros((h, w), dtype=np.uint32)
alpha = 0.1  # Frame averaging parameter

# Read class names from a file
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Main loop for processing video frames
while cap.isOpened():
    # Read a frame from the video
    rt, video = cap.read()

    # Create a copy of the original frame for visualization purposes
    vid = video.copy()

    # Resize the frames for consistent dimensions
    video = cv2.resize(video, (800, 700))
    vid = cv2.resize(vid, (800, 700))

    # Predict objects in the video frame using the YOLO model
    result = model.predict(video)[0]
    bboxs = result.prediction.bboxes_xyxy
    confidence = result.prediction.confidence
    labels = result.prediction.labels

    # Process each detected object
    for (bboxs, confidence, labels) in zip(bboxs, confidence, labels):
        x1, y1, x2, y2 = np.array((bboxs))
        x1, y1, x2, y2 = int(bboxs[0]), int(bboxs[1]), int(bboxs[2]), int(bboxs[3])
        confidence = math.ceil(confidence * 100)
        labels = int(labels)
        classdetect = classnames[labels]

        # Draw bounding box on the video frame
        cv2.rectangle(video, (x1, y1), (x2, y2), (0, 255, 0), 2)
        globalimagearray[y1:y2, x1:x2] += 1

        # Add class name as text if confidence is above 20
        if confidence > 20:
            cvzone.putTextRect(video, f'{classdetect}', [x1 + 8, y1 - 12], thickness=2, scale=1.5)
            cvzone.putTextRect(vid, 'YOLO-NAS HEATMAP', (50, 50))
            cvzone.putTextRect(vid, 'INPUT VIDEO', (90, 670))

    # Normalize and blur the heatmap
    globalarraynorm = (globalimagearray - globalimagearray.min()) / (
            globalimagearray.max() - globalimagearray.min() + 1e-8) * 255
    globalarraynorm = globalarraynorm.astype('uint8')
    globalarraynorm = cv2.GaussianBlur(globalarraynorm, (25, 25), 0)

    # Apply color map to create heatmap visualization
    heatmap = cv2.applyColorMap(globalarraynorm, cv2.COLORMAP_HOT)
    heatmap = cv2.resize(heatmap, (800, 700))

    # Resize video frame for heatmap averaging
    alpha_frame = cv2.resize(video, (w, h))
    alpha_frame_gray = cv2.cvtColor(alpha_frame, cv2.COLOR_BGR2GRAY)

    # Perform frame averaging for heatmap smoothing
    globalimagearray = alpha * alpha_frame_gray + (1 - alpha) * globalimagearray

    # Create a final image by blending the heatmap and original video frame
    finalimg = cv2.addWeighted(heatmap, 0.7, video, 0.3, 0)

    # Stack images for display
    output_frames = cvzone.stackImages([vid, video, finalimg, heatmap], 2, 0.70)

    # Display the stacked images
    cv2.imshow('frame', output_frames)

    # Wait for a key press
    cv2.waitKey(1)

# Release resources
cap.release()
cv2.destroyAllWindows()
