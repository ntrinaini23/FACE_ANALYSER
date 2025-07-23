from deepface import DeepFace
import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to improve performance
    resized_frame = cv2.resize(frame, (640, 480))

    try:
        # Analyze the frame
        result = DeepFace.analyze(
            resized_frame,
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False
        )

        # Show results on frame
        if result:
            face_data = result[0] if isinstance(result, list) else result
            age = face_data.get("age", "N/A")
            gender = face_data.get("gender", "N/A")
            emotion = face_data.get("dominant_emotion", "N/A")

            cv2.putText(resized_frame, f'Age: {age}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(resized_frame, f'Gender: {gender}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(resized_frame, f'Emotion: {emotion}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            print("No face detected")

    except Exception as e:
        print("Face detection failed:", e)

    # Show the result frame
    cv2.imshow("Real-Time Face Analysis", resized_frame)

    # Press 'q' or ESC to exit
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
