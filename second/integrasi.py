import ctypes
import sys
import cv2
import numpy as np
from library.sort import Sort
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

# Load model yolo untuk upper body
net_upper_body = cv2.dnn.readNet("../cfg/yolov4-tiny-custom_final.weights", "../cfg/yolov4-tiny-custom.cfg")
layer_names_upper = net_upper_body.getLayerNames()
output_layers_upper = [layer_names_upper[i - 1] for i in net_upper_body.getUnconnectedOutLayers()]

# Load model yolo untuk wajah
net_face = cv2.dnn.readNet("../cfg/yolov4-tiny-custom_final_face.weights", "../cfg/yolov4-tiny-customFace.cfg")
layer_names_face = net_face.getLayerNames()
output_layers_face = [layer_names_face[i - 1] for i in net_face.getUnconnectedOutLayers()]

# Load label kelas
with open("../cfg/obj.names", "r") as f:
    classes_upper = [line.strip() for line in f.readlines()]

with open("../cfg/objFaces.names", "r") as f:
    classes_face = [line.strip() for line in f.readlines()]

# inisiasi tracking sort & kinect
tracking_upper_body = Sort()
tracking_face = Sort()
cap = cv2.VideoCapture(1)
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

while True:
    # # tangkap frame rgb
    # rawFrame = kinect.get_last_color_frame()
    # frame = rawFrame.reshape((1080,1920, 4)).astype(np.uint8)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    #
    # #tangkap frame depth
    # rawFrame2 = kinect.get_last_depth_frame()
    # depthFrame = rawFrame2.reshape((424, 512)).astype(np.uint16)
    # depthFrameNormal = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    ret, frame = cap.read()
    if ret is None:
        break
    height, width, channels = frame.shape

    # Preprocessing untuk deteksi upper body
    blob_upper = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net_upper_body.setInput(blob_upper)
    outs_upper = net_upper_body.forward(output_layers_upper)

    class_ids_upper = []
    confidences_upper = []
    boxes_upper = []

    # Proses output upper body
    for out in outs_upper:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Threshold confidence
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes_upper.append([x, y, w, h])
                confidences_upper.append(float(confidence))
                class_ids_upper.append(class_id)

    # terapkan NMS untuk upper body
    indexes_upper = cv2.dnn.NMSBoxes(boxes_upper, confidences_upper, 0.5, 0.4)
    detections_upper = []

    # Menerapkan bounding box untuk deteksi upper body
    if len(indexes_upper) > 0:
        for i in indexes_upper.flatten():
            x, y, w, h = boxes_upper[i]
            confidence = confidences_upper[i]
            detections_upper.append([x, y, x + w, y + h, confidence])

    # Terapkan SORT tracking pada upper body
    if len(detections_upper) > 0:
        tracked_upper_body = tracking_upper_body.update(np.array(detections_upper))
    else:
        tracked_upper_body = []
#====================================================
    # Deteksi wajah hanya di area upper body
    for obj in tracked_upper_body:
        x1, y1, x2, y2, track_id = map(int, obj[:5])

        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > width: x2 =width
        if y2 > height: y2 = height

        if(x2 - x1 > 0) and (y2 - y1 > 0):
            upper_body_frame = frame[y1:y2, x1:x2]

            # Preprocessing untuk deteksi wajah di area upper body
            blob_face = cv2.dnn.blobFromImage(upper_body_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net_face.setInput(blob_face)
            outs_face = net_face.forward(output_layers_face)

        class_ids_face = []
        confidences_face = []
        boxes_face = []

        for out in outs_face:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * (x2 - x1))
                    center_y = int(detection[1] * (y2 - y1))
                    w = int(detection[2] * (x2 - x1))
                    h = int(detection[3] * (y2 - y1))
                    x = int(center_x - w / 2) + x1
                    y = int(center_y - h / 2) + y1

                    boxes_face.append([x, y, w, h])
                    confidences_face.append(float(confidence))
                    class_ids_face.append(class_id)

        # terapkan NMS untuk wajah
        indexes_face = cv2.dnn.NMSBoxes(boxes_face, confidences_face, 0.5, 0.4)
        detections_face = []

        if len(indexes_face) > 0:
            for i in indexes_face.flatten():
                x, y, w, h = boxes_face[i]
                confidence = confidences_face[i]
                detections_face.append([x, y, x + w, y + h, confidence])

        # Terapkan SORT tracking pada wajah
        if len(detections_face) > 0:
            tracked_face = tracking_face.update(np.array(detections_face))
        else:
            tracked_face = []

        # Gambar bounding box upper body dan wajah
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # bounding box upper body
        for obj in tracked_face:
            x1_face, y1_face, x2_face, y2_face, track_id_face = map(int, obj[:5])
            label = f"Face ID: {int(track_id_face)}"
            cv2.rectangle(frame, (x1_face, y1_face), (x2_face, y2_face), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1_face, y1_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("Upper Body and Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kinect.release()
cv2.destroyAllWindows()
