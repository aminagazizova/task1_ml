from transformers import pipeline
import cv2
from PIL import Image

detector = pipeline("object-detection", model="facebook/detr-resnet-50")

video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Не удалось открыть видео: {video_path}")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_detected.mp4', fourcc, fps, (width, height))

frame_count = 0

print("Обработка видео... (может занять несколько секунд)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 10 != 0:
        out.write(frame)
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    results = detector(frame_pil)

    for obj in results:
        label = obj['label']
        score = obj['score']
        box = obj['box']

        x1, y1, x2, y2 = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    out.write(frame)

    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Обработка завершена! Видео сохранено как output_detected.mp4")
