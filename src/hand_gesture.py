import cv2
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start video capture
cap = cv2.VideoCapture(0)

# Set resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def classify_hand_gesture(landmarks):
    """Classify hand gestures based on finger positions"""
    thumb_tip = landmarks[4].y
    thumb_ip = landmarks[3].y
    index_tip = landmarks[8].y
    middle_tip = landmarks[12].y
    ring_tip = landmarks[16].y
    pinky_tip = landmarks[20].y
    wrist = landmarks[0].y

    # Open palm: All fingers are extended
    if all(tip < wrist for tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "Hello, friend"

    # Thumbs up: Thumb extended, other fingers curled
    if thumb_tip < thumb_ip and all(tip > wrist for tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "Hello, friend"

    # Closed fist: All fingers curled
    if all(tip > wrist for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]):
        return "We welcome others with palm hands, not with closed fists"

    return None  # No recognized gesture

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    message = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            message = classify_hand_gesture(hand_landmarks.landmark)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display message on screen
    if message:
        cv2.putText(frame, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
