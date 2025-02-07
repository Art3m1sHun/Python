import cv2
import mediapipe as mp
import time
import turtle
from nltk.ccg.combinator import forwardTConstraint
from tensorflow.python.ops.signal.shape_ops import frame

# Initialize Turtle
t = turtle.Turtle()
t.color("Blue")

# Tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Video Capture
cap = cv2.VideoCapture(0)
threshold = 30

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("No frame captured.")
        break

    image = cv2.flip(image, 1)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Khởi tạo cờ cử chỉ
    turn_left = turn_right = backward_lui = forward = False

    if results.multi_hand_landmarks:
        left_hand_detected = right_hand_detected = False

        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, c = image.shape

            # Lấy tọa độ các điểm mốc
            thumb_x = int(hand_landmarks.landmark[4].x * w)
            thumb_y = int(hand_landmarks.landmark[4].y * h)
            index_x = int(hand_landmarks.landmark[8].x * w)
            index_y = int(hand_landmarks.landmark[8].y * h)

            #Tính toán khoảng cách giữa ngón tay trỏ và ngón tay trái
            distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5

            # Xác định bàn tay nào được phát hiện
            handedness = results.multi_handedness[hand_index].classification[0].label

            # Kiểm tra cử chỉ từng bàn tay
            if handedness == "Left":
                left_hand_detected = True
                left_hand_distance = distance
                if distance < threshold:
                    turn_left = True

            elif handedness == "Right":
                right_hand_detected = True
                right_hand_distance = distance
                if distance < threshold:
                    turn_right = True

        # Kiểm tra điều kiện cả hai tay thực hiên tiến lên và đi lu
        if left_hand_detected and right_hand_detected:
            if left_hand_distance < threshold and right_hand_distance < threshold:
                backward_lui = True
            elif left_hand_distance > threshold and right_hand_distance > threshold:
                forward = True

    # Điều khiên con trỏ
    if backward_lui:
        t.back(50)
        print("Back")
        time.sleep(0.5)
    elif forward:
        t.forward(50)
        print("Moving Forward")
        time.sleep(0.5)
    elif turn_left:
        t.left(50)
        print("Turning Left")
        time.sleep(0.5)
    elif turn_right:
        t.right(50)
        print("Turning Right")
        time.sleep(0.5)

    # Hiển thị thời gian
    text = f"time: {time.strftime('%H:%M:%S')}"
    cv2.putText(image, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Gesture Control', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
