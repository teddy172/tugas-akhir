import cv2
import numpy as np
from pykinect2 import PyKinectRuntime, PyKinectV2
from library.sort import Sort

# Load model YOLO
net = cv2.dnn.readNet("../cfg/yolov4-tiny-custom_final.weights", "..//cfg/yolov4-tiny-custom.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load label kelas COCO
with open("../cfg/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Inisiasi Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

# Inisiasi SORT tracking
tracking = Sort()

while True:
    # Ambil frame depth
    if kinect.has_new_depth_frame():
        depth_frame = kinect.get_last_depth_frame()
        depth_frame = depth_frame.reshape((424, 512)).astype(np.uint16)
        depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Ambil frame RGB
    if kinect.has_new_color_frame():
        color_frame = kinect.get_last_color_frame()
        color_frame = color_frame.reshape((1080, 1920, 4)).astype(np.uint8)
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)  # Ubah ke BGR agar kompatibel dengan OpenCV
        height, width, channels = color_frame.shape

        # Preprocessing YOLO
        blob = cv2.dnn.blobFromImage(color_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        #proses output YOLO
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

        #(NMS)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                detections.append([x, y, x + w, y + h, confidence])

        # Update SORT
        if len(detections) > 0:
            tracked_objects = tracking.update(np.array(detections))
        else:
            tracked_objects = []

        # Gambar bounding box hasil sort pada frame
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj[:5])
            label = f"ID: {int(track_id)}"
            color = (0, 0, 255)
            cv2.rectangle(color_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(color_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            depth_x = int(center_x * (512 / 1920))
            depth_y = int(center_y * (424 / 1080))

            # Ambil nilai depth pada titik tengah bounding box
            depth_value = depth_frame[depth_y, depth_x]
            distance_in_meters = depth_value * 0.001
            print(f"jarak: {depth_value}")

    # Gabungkan dan tampilkan frame RGB (dengan bounding box) dan depth jika keduanya tersedia
    if 'color_frame' in locals() and 'depth_frame' in locals():
        # Tampilkan frame dengan bounding box dan ID tracking
        resized_frame = cv2.resize(color_frame, (640, 360))
        resized_depth = cv2.resize(depth_frame, (640, 360))

        # Gabungkan frame RGB dan depth secara berdampingan
        combined_frame = np.hstack((resized_frame, cv2.cvtColor(resized_depth, cv2.COLOR_GRAY2BGR)))
        cv2.imshow("Real-Time Object Detection and Depth", combined_frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Akhiri Kinect dan jendela OpenCV
kinect.close()
cv2.destroyAllWindows()
