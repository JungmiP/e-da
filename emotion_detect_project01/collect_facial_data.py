import copy
import csv
import itertools

from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def preprocess_landmark(landmark_list):
    tmp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for idx, landmark_point in enumerate(tmp_landmark_list):
        if idx == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        tmp_landmark_list[idx][0] -= base_x
        tmp_landmark_list[idx][1] -= base_y

    tmp_landmark_list = list(itertools.chain.from_iterable(tmp_landmark_list))
    max_value = max(list(map(abs, tmp_landmark_list)))

    def normalize_(n):
        return n / max_value

    tmp_landmark_list = list(map(normalize_, tmp_landmark_list))

    return tmp_landmark_list

def save_csv(landmark_list):
    csv_path = "./facemesh_dataset/keypoint.csv"
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([landmark_list])



# cap_device = 0
# cap_width = 1920
# cap_height = 1080

# 웹캠의 화면을 가져오는 객체
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

# 얼굴과 얼굴의 각 포인트 위치 정보 탐지하는 객체
mp_face_mesh = mp.solutions.face_mesh
# 인식한 landmark를 그릴 객체
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles




with mp_face_mesh.FaceMesh(
                            max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.7, #얼굴 인식에 필요한 최소 신뢰도 점수
                            min_tracking_confidence=0.5 #얼굴 추적의 최소 신뢰도 점수
) as face_mesh:

    while cap.isOpened()==True:
        success, image = cap.read()

        if success == False:
            continue

        # 이미지를 거울 모드로 변경
        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # BGR 이미지를 RGB로 변경
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #image.flags.writeable = False # 이미지 다시쓰기 false
        results = face_mesh.process(image)
        #image.flags.writeable = True
        # results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks is not None:

            for face_landmarks in results.multi_face_landmarks:

                # 탐지한 얼굴 landmark 그리기
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=1),
                #     connection_drawing_spec= mp_drawing_styles.get_default_face_mesh_tesselation_style()
                # )
                #
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0, 255), thickness=2, circle_radius=2),
                #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                #
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_IRISES,
                #     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())


                landmark_list = calc_landmark_list(debug_image, face_landmarks)
                preprocessed_landmark_list = preprocess_landmark(landmark_list)
                save_csv(preprocessed_landmark_list)


        # 웹캠 화면 화면에 출력
        cv2.imshow('webcam_window01', image)

        # 사용자가 'q'를 입력하면 종료
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()