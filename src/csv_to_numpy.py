import pandas as pd
import numpy as np
from PIL import Image

from constants import *
import cv2

classifier = cv2.CascadeClassifier(CLASSIFIER_PATH)


def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    gray_border = np.zeros((150, 150), np.uint8)
    gray_border[:, :] = 200
    gray_border[((150 // 2) - (SIZE_FACE // 2)):((150 // 2) + (SIZE_FACE // 2)),
    ((150 // 2) - (SIZE_FACE // 2)):((150 // 2) + (SIZE_FACE // 2))] = image
    image = gray_border

    faces = classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    # None is we don't found an image
    if not len(faces) > 0:
        # print "No hay caras"
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size

    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None
    print(image.shape)
    return image


def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d


def flip_image(image):
    return cv2.flip(image, 1)


def data_to_image(data):
    # print data
    data_image = np.fromstring(str(data), dtype=np.uint8, sep=' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy()
    data_image = format_image(data_image)
    return data_image


dataSet = pd.read_csv(DATA_SET_FILE)

training_labels = []
training_image = []
test_labels = []
test_image = []
index = 1
total = dataSet.shape[0]

for index, row in dataSet.iterrows():
    usage = row['Usage']
    emotion = emotion_to_vec(row['emotion'])
    image = data_to_image(row['pixels'])
    if image is not None:
        if usage == 'Training':
            training_labels.append(emotion)
            training_image.append(image)
        elif usage == 'PublicTest':
            test_labels.append(emotion)
            test_image.append(image)
        elif usage == 'PrivateTest':
            test_labels.append(emotion)
            test_image.append(image)

            # labels.append(emotion)
            # images.append(flip_image(image))
    else:
        print("Error")
    index += 1
    print("Progress: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

print("Total: " + str(len(training_image) + len(test_image)))
np.save(TRAINING_SET, training_image)
np.save(TRAINING_LABELS, training_labels)
np.save(TEST_SET, test_image)
np.save(TEST_LABELS, test_labels)
