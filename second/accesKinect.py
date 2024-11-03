import ctypes
import sys
import numpy as np
import cv2

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

# Inisialisasi Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Body)

# Fungsi untuk menggambar skeleton (garis antar joint)
def draw_skeleton(frame, joints, joint_points):
    bones = [
        (PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck),
        (PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder),
        (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid),
        (PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase),
        (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft),
        (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight),
        (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft),
        (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight),
        (PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft),
        (PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft),
        (PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft),
        (PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight),
        (PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight),
        (PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight),
        (PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft),
        (PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft),
        (PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft),
        (PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight),
        (PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight),
        (PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight)
    ]

    for bone in bones:
        joint1 = bone[0]
        joint2 = bone[1]

        if joints[joint1].TrackingState == PyKinectV2.TrackingState_Tracked and \
           joints[joint2].TrackingState == PyKinectV2.TrackingState_Tracked:
            point1 = (int(joint_points[joint1].x), int(joint_points[joint1].y))
            point2 = (int(joint_points[joint2].x), int(joint_points[joint2].y))
            cv2.line(frame, point1, point2, (0, 255, 0), 5)

def main():
    while True:
        # Periksa apakah frame warna tersedia
        if kinect.has_new_color_frame():
            # Ambil frame warna
            color_frame = kinect.get_last_color_frame()
            color_image = color_frame.reshape((1080, 1920, 4)).astype(np.uint8)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGBA2RGB)

            # Tampilkan frame warna di jendela
            cv2.imshow('Kamera Warna (RGB)', color_image)

        # Periksa apakah frame kedalaman tersedia
        if kinect.has_new_depth_frame():
            # Ambil frame kedalaman
            depth_frame = kinect.get_last_depth_frame()
            depth_image = depth_frame.reshape((424, 512)).astype(np.uint16)

            # Normalisasi frame kedalaman untuk ditampilkan
            depth_image_normalized = np.uint8(depth_image / np.max(depth_image) * 255)
            depth_image_bgr = cv2.cvtColor(depth_image_normalized, cv2.COLOR_GRAY2BGR)

            # Periksa apakah ada body yang terdeteksi
            if kinect.has_new_body_frame():
                bodies = kinect.get_last_body_frame()

                if bodies is not None:
                    for i in range(0, kinect.max_body_count):
                        body = bodies.bodies[i]
                        if body.is_tracked:
                            joints = body.joints
                            joint_points = {}

                            for joint in range(PyKinectV2.JointType_Count):
                                joint_position = joints[joint].Position
                                depth_point = kinect._mapper.MapCameraPointToDepthSpace(joint_position)
                                joint_points[joint] = depth_point

                            # Gambar skeleton di atas frame kedalaman
                            draw_skeleton(depth_image_bgr, joints, joint_points)

            # Tampilkan frame kedalaman di jendela
            cv2.imshow('Sensor Kedalaman dengan Skeleton', depth_image_bgr)

        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Hentikan Kinect dan tutup jendela
    kinect.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
