import cv2
import numpy as np
from library.sort import Sort

# inisiasi var
frame_counter = 0
saved_stage1 = False
saved_stage2 = False
number_image = 25

# Load model YOLO
net = cv2.dnn.readNet("../cfg/yolov4-tiny-custom_final.weights", "../cfg/yolov4-tiny-custom.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]#mengembalikan layer output

# Load label kelas
with open("../cfg/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Akses kamera laptop
cap = cv2.VideoCapture(0)

# Inisiasi tracking dengan SORT
tracking = Sort()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, channels = frame.shape

    # Preprocessing gambar untuk YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Memproses output dari YOLO
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

    # Terapan NMS untuk menghilangkan bounding box yang tidak diperlukan
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detections = []

    # Mempersiapkan format deteksi untuk SORT
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            detections.append([x, y, x + w, y + h, confidence])#kordinat xy, kanan bawah, kiri atas

    # update sort tracking dengan array bb objek
    if len(detections) > 0:
        tracked_object = tracking.update(np.array(detections))
    else:
        tracked_object = []

    # Membuat bounding box dan mendapatkan ROI baju================================
    for obj in tracked_object:
        x1, y1, x2, y2, track_id = map(int, obj[:5])
        label = f"ID: {int(track_id)}"
        color = (0, 255, 0)  # Warna untuk bounding box
        color2 = (0, 0, 255)

        roi_upper_body = frame[y1:y2, x1:x2]

        # Mendapatkan bagian bawah dari ROI sebagai ROI baju
        if roi_upper_body.size > 0:
            roi_warna = roi_upper_body.copy()
            height_roi, width_roi, _ = roi_warna.shape

            # Ambil sepertiga bagian bawah dari ROI upper body untuk area baju
            part_height = height_roi // 3

            objek_ada=True
            reduction = 40
            reduction = min(reduction, width_roi // 2)
            x_baju = x1 + reduction
            y_baju = y1 + 2 * part_height
            w_baju = width_roi - 2 * reduction
            h_baju = part_height  # Tinggi 1/3 upper body
            roi_baju_crop = frame[y_baju:y_baju + h_baju, x_baju:x_baju + w_baju]

            if not saved_stage1:
                cv2.imwrite(f"../gambar/cropped_baju_stage1_{track_id}.jpg", roi_baju_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                saved_stage1 = True
                print(f"Tahap 1: Gambar objek {track_id} disimpan.")

            elif frame_counter == 15 and not saved_stage2:
                cv2.imwrite(f"../gambar/cropped_baju_stage2_{track_id}.jpg", roi_baju_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                saved_stage2 = True
                print(f"Tahap 2: Gambar objek {track_id} disimpan pada frame ke-15.")

            elif frame_counter >= 25:
                cv2.imwrite(f"../gambar/cropped_baju_stage3_frame_{number_image}_{track_id}.jpg", roi_baju_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                print(f"Tahap 3: Gambar objek {track_id} disimpan dari frame ke-{number_image}.")
                number_image +=1

            # Gambar bounding box di frame utama
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (x_baju, y_baju), (x_baju + w_baju, y_baju + h_baju), color2, 2)
            frame_counter += 1
        else:
            frame_counter = 0

    cv2.imshow("Real-Time Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from library.sort import Sort
#
# # Inisiasi var
# frame_counter = 0
# saved_stage1 = False
# saved_stage2 = False
# number_image = 25
#
# # Load model YOLO
# net = cv2.dnn.readNet("../cfg/yolov4-tiny-custom_final.weights", "../cfg/yolov4-tiny-custom.cfg")
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] #mengembalikan layer output
#
# # Load label kelas
# with open("../cfg/coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
#
# # Akses kamera laptop
# cap = cv2.VideoCapture(1)
#
# # Inisiasi tracking dengan SORT
# tracking = Sort()
#
# # Fungsi untuk menghitung kesamaan warna menggunakan HSV Histogram
# def hitung_similarity_hsv(path1, path2, path3):
#     bins = [50, 60]
#     ranges = [0, 180, 0, 256]
#     channels = [0, 1]
#
#     # Membaca dan mengubah gambar ke HSV
#     gambar_hsv1 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2HSV)
#     gambar_hsv2 = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2HSV)
#     gambar_hsv3 = cv2.cvtColor(cv2.imread(path3), cv2.COLOR_BGR2HSV)
#
#     # Menghitung histogram
#     hist_tahap1 = cv2.calcHist([gambar_hsv1], channels, None, bins, ranges)
#     hist_tahap2 = cv2.calcHist([gambar_hsv2], channels, None, bins, ranges)
#     hist_tahap3 = cv2.calcHist([gambar_hsv3], channels, None, bins, ranges)
#
#     # Normalisasi histogram
#     cv2.normalize(hist_tahap1, hist_tahap1)
#     cv2.normalize(hist_tahap2, hist_tahap2)
#     cv2.normalize(hist_tahap3, hist_tahap3)
#
#     # Menghitung kesamaan menggunakan metode Histogram Comparison
#     method = cv2.HISTCMP_CORREL
#     er1 = cv2.compareHist(hist_tahap1, hist_tahap3, method)
#     er2 = cv2.compareHist(hist_tahap2, hist_tahap3, method)
#     total_similarity = (er1 + er2) / 2
#
#     print(f"Similarity dengan Stage 1: {er1}")
#     print(f"Similarity dengan Stage 2: {er2}")
#     print(f"Rata-rata Similarity: {total_similarity}")
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     height, width, channels = frame.shape
#
#     # Preprocessing gambar untuk YOLO
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#
#     # Memproses output dari YOLO
#     class_ids = []
#     confidences = []
#     boxes = []
#
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.3:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#
#     # Terapan NMS untuk menghilangkan bounding box yang tidak diperlukan
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#     detections = []
#
#     # Mempersiapkan format deteksi untuk SORT
#     if len(indexes) > 0:
#         for i in indexes.flatten():
#             x, y, w, h = boxes[i]
#             confidence = confidences[i]
#             detections.append([x, y, x + w, y + h, confidence]) #kordinat xy, kanan bawah, kiri atas
#
#     # update sort tracking dengan array bb objek
#     if len(detections) > 0:
#         tracked_object = tracking.update(np.array(detections))
#     else:
#         tracked_object = []
#
#     # Membuat bounding box dan mendapatkan ROI baju================================
#     for obj in tracked_object:
#         x1, y1, x2, y2, track_id = map(int, obj[:5])
#         label = f"ID: {int(track_id)}"
#         color = (0, 255, 0)  # Warna untuk bounding box
#         color2 = (0, 0, 255)
#
#         roi_upper_body = frame[y1:y2, x1:x2]
#
#         # Mendapatkan bagian bawah dari ROI sebagai ROI baju
#         if roi_upper_body.size > 0:
#             roi_warna = roi_upper_body.copy()
#             height_roi, width_roi, _ = roi_warna.shape
#
#             # Ambil sepertiga bagian bawah dari ROI upper body untuk area baju
#             part_height = height_roi // 3
#
#             reduction = 40
#             reduction = min(reduction, width_roi // 2)
#             x_baju = x1 + reduction
#             y_baju = y1 + 2 * part_height
#             w_baju = width_roi - 2 * reduction
#             h_baju = part_height  # Tinggi 1/3 upper body
#             roi_baju_crop = frame[y_baju:y_baju + h_baju, x_baju:x_baju + w_baju]
#
#             # Simpan gambar sesuai tahap
#             if not saved_stage1:
#                 cv2.imwrite("../gambar/cropped_baju_stage1.jpg", roi_baju_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
#                 saved_stage1 = True
#                 print("Tahap 1: Gambar disimpan.")
#
#             elif frame_counter == 15 and not saved_stage2:
#                 cv2.imwrite("../gambar/cropped_baju_stage2.jpg", roi_baju_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
#                 saved_stage2 = True
#                 print("Tahap 2: Gambar disimpan pada frame ke-15.")
#
#             elif frame_counter >= 25:
#                 cv2.imwrite(f"../gambar/cropped_baju_stage3_frame_{number_image}.jpg", roi_baju_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
#                 print(f"Tahap 3: Gambar disimpan dari frame ke-{number_image}.")
#                 number_image += 1
#
#                 # Hitung kesamaan HSV setelah tahap 3
#                 hitung_similarity_hsv("../gambar/cropped_baju_stage1.jpg",
#                                       "../gambar/cropped_baju_stage2.jpg",
#                                       f"../gambar/cropped_baju_stage3_frame_{number_image-1}.jpg")
#
#             # Gambar bounding box di frame utama
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#             cv2.rectangle(frame, (x_baju, y_baju), (x_baju + w_baju, y_baju + h_baju), color2, 2)
#             frame_counter += 1
#         else:
#             frame_counter = 0
#
#     cv2.imshow("Real-Time Object Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
