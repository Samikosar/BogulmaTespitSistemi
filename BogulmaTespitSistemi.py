from ultralytics import YOLO
import cv2
model = YOLO('best.pt')
video_path = 'deneme.mp4'
cap = cv2.VideoCapture(video_path)

frame_skip = 1
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:

        results = model.track(frame, persist=True)

        frame = results[0].plot()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print("Video işleme tamamlandı")
