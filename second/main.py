import cv2
import numpy as np
from library.sort import Sort
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

# Load model yolo
net = cv2.dnn.readNet("../cfg/yolov4-tiny-custom_final.weights", "../cfg/yolov4-tiny-custom.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load Common Objects in Context label kelas
with open("../cfg/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# akses kamera laptop
# cap = cv2.VideoCapture(1)
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

# Counter untuk menyimpan foto dengan nama yang berbeda
frame_count = 0

#inisiasi tracking library
tracking = Sort()

while True:
    timer_start = cv2.getTickCount()
    # tangkap frame per frame
    # ret, frame = cap.read()
    # if ret is None:
    #     break
    # height, width, channels = frame.shape

    # tangkap frame rgb
    rawFrame = kinect.get_last_color_frame()
    frame = rawFrame.reshape((1080,1920, 4)).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    #tangkap frame depth
    # rawFrame2 = kinect.get_last_depth_frame()
    # depthFrame = rawFrame2.reshape((424, 512)).astype(np.uint16)
    # frame = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # frame = cv2.merge([frame, frame, frame])  # Menambahkan dua saluran tambahan

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

    # terapkan NMS untuk menghilangkan bounding box yang tidak diperlukan
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detections = []

    # terapkan bounding box pada gambar
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            # rubah format deteksi untuk library
            detections.append([x, y, x + w, y + h, confidence])

    #terapkan library
    if len(detections) > 0:
        tracked_object = tracking.update(np.array(detections))
    else:
        tracked_object = []
#==========================================================
    #bentuk bounding box kembali
    for obj in tracked_object:
        x1, y1, x2, y2, track_id = obj
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"ID: {int(track_id)}"
        color = (0, 255, 0)  # warna untuk bounding box

        #tetapkan bidang bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#===========================================================
    # Hitung FPS
    timer_end = cv2.getTickCount()
    time_per_frame = (timer_end - timer_start) / cv2.getTickFrequency()
    fps = 1.0 / time_per_frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kinect.release()
cv2.destroyAllWindows()