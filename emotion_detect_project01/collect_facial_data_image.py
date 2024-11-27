import csv
import copy
import itertools
import os
import cv2
import mediapipe as mp

def encode_label(label_name, category):
    for i in category:
        if i == label_name:
            return category.index(i)

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

def save_csv(number, landmark_list):
    if 0 <= number <= 5:
        csv_path = "./dataset/keypoint.csv"
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

root = ""
IMAGE_FILES = []
category = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
for path, subdirs, files in os.walk(root):
    for name in files:
        IMAGE_FILES.append(os.path.join(path, name))
use_brect = True

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    static_image_mode=True )

for idx, file in enumerate(IMAGE_FILES):
    label_name = file.rsplit("/", -1)[-1]
    label_name = label_name.rsplit("\\", 1)[0]
    label = encode_label(label_name, category)
    image = cv2.imread(file)
    image = cv2.filp(image, 1)
    debug_image = copy.deepcopy(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            landmark_list = calc_landmark_list(debug_image, face_landmarks)
            preprocessed_landmark_list = preprocess_landmark(landmark_list)
            save_csv(label, preprocessed_landmark_list)