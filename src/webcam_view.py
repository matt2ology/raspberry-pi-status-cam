import cv2

import logging
FORMAT = '[%(asctime)s]-[%(funcName)s]-[%(levelname)s] - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT
)

# Open the first available camera (0 is usually the default USB camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    logging.error("Could not open video stream")
    exit()

# Loop to continuously get frames
while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to get frame")
        break

    cv2.imshow("Webcam Feed", frame)  # Display the frame in a window

    # Wait for the 'q' key to be pressed to closed the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()