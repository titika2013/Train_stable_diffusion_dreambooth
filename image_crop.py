import os

import numpy as np
import cv2
from deepface import DeepFace
import glob
# import filetype # do in future

def resize(image, size):
    # If you are enlarging the image use INTER_CUBIC
    # If you are shrinking the image use INTER_AREA
    height, width, channels = image.shape
    if height + width > sum(size):
        resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    return resized


#
#
def img_process(source_photo_dir="fresh_photos", dst_photo_dir="my_photos", crop_size=512, face_detection=True,
                key_of_photo="MUYN"):
    size = (crop_size, crop_size)
    count = 0
    backends = [
        'opencv',
        'ssd',
        'dlib',
        'mtcnn',
        'retinaface',
        'mediapipe'
    ]
    if not os.path.exists(dst_photo_dir):
        os.mkdir(dst_photo_dir)
    for count, item in enumerate(glob.iglob(os.path.join(source_photo_dir, "*"))):  # *.jpg
        try:
            img = cv2.imread(item)
            if face_detection:
                pass
                face = DeepFace.detectFace(img_path=img, target_size=size, detector_backend=backends[4],
                                           enforce_detection=False)
                if np.mean(face) < 1e-3:
                    continue
            resized_img = resize(img, size)
            dst = f"{key_of_photo}_({str(count)}).jpg"
            cv2.imwrite(os.path.join(dst_photo_dir, dst), resized_img)
        except:
            print("something went wrong")
            continue
    if count == 0:
        print("Error not images")
    else:
        print(f'[1;32m \n Image crop Done for {count + 1} images')
