import os
import argparse
import numpy as np
import cv2
from deepface import DeepFace
import glob


# # @markdown Where are your photos
# source_photo_dir = "/content/gdrive/MyDrive/fresh_photos"  # @param {type:"string"}
#
# # @markdown Where to send your photos
# dst_photo_dir = "/content/gdrive/MyDrive/my_photos"  # @param {type:"string"}
#
# # @markdown What is the name of your prompt
# photo_name_start = "tigran"  # @param {type:"string"}
#
# # @markdown Photos with only faces?
# face_detection = True  # @param {type:"boolean"}
#

#
#
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
def process(source_photo_dir="fresh_photos", dst_photo_dir="my_photos", crop_size=512, face_detection=True,
            key_of_photo="MUYN"):
    size = (crop_size, crop_size)
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
            dst = f"{key_of_photo}({str(count)}).jpg"
            cv2.imwrite(os.path.join(dst_photo_dir, dst), resized_img)
        except:
            print("something went wrong")
            continue


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--image_path', type=str, default='fresh_photos', help='path to init folder image')
    parser.add_argument('--save_image_path', type=str, default='my_photos', help='path to resized images')
    parser.add_argument('--key_name', type=str, default='MUYN', help='renamed image key')
    parser.add_argument('--crop_size', type=int, default=512, help='Crop size')
    parser.add_argument('--face_finder', type=bool, default=True, help="crop and save only images with faces")
    args = parser.parse_args()
    image_path = args.image_path
    save_image_path = args.save_image_path
    crop_size = args.crop_size
    need_face_find = args.face_finder
    img_names = args.key_name
    process(source_photo_dir=image_path, dst_photo_dir=save_image_path, crop_size=crop_size,
            face_detection=need_face_find, key_of_photo=img_names)
