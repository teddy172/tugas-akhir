import cv2
import numpy as np
from library.sort import Sort

# Load model yolo
net = cv2.dnn.readNet("../cfg/yolov4-tiny-custom_final_face.weights", "../cfg/yolov4-tiny-customFace.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load Common Objects in Context label kelas
with open("../cfg/objFaces.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)
frame_count = 0
tracking = Sort()

while True:
    timer_start = cv2.getTickCount()
    # tangkap frame per frame
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # preprocessing gambar set ukuran, ubah ke bgr
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # memproses output
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    detections = []

    # terapkan bounding box pada gambar
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            # rubah format deteksi untuk sort
            detections.append([x, y, x + w, y + h, confidence])

    #terapkan sort
    if len(detections) > 0:
        tracked_object = tracking.update(np.array(detections))
    else:
        tracked_object = []

    #bentuk bounding box
    for obj in tracked_object:
        x1, y1, x2, y2, track_id = obj
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = class_ids[i]
        class_name = classes[class_id]
        label = f"ID: {class_name}"
        color = (0, 255, 0)  # warna untuk bounding box

        #tetapkan bidang bounding box
        # Gunakan bounding box hasil dari Sort
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # frame = cv2.resize(frame, (416, 416))
    # Hitung FPS
    timer_end = cv2.getTickCount()
    time_per_frame = (timer_end - timer_start) / cv2.getTickFrequency()
    fps = 1.0 / time_per_frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
