# JSON Dumping

import cv2
import sys
import numpy as np
import argparse
import datetime
import os
import glob
import json
from retinaface import RetinaFace
from tqdm.auto import tqdm

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def parse_args():
    parser = argparse.ArgumentParser(description='Inference RetinaFace')

    # input json file path (ex. '/content/drive/MyDrive/rippleai/input.json')
    parser.add_argument('input_file',
                        help='input json file',
                        type=str)

    # output json file path (ex. '/content/drive/MyDrive/rippleai/output.json')                
    parser.add_argument('output_file',
                        help='output json file',
                        type=str)

    # model directory path
    parser.add_argument('--model_path',
                        help='model directory path',
                        default='./model/R50',
                        type=str)

    # threshold (0 ~ 1)
    parser.add_argument('--thr',
                        help='threshold (0 ~ 1)',
                        default=0.8,
                        type=float)

    args = parser.parse_args()
    return args


def main():
  args = parse_args()
  thresh = args.thr
  scales = [1024, 1980]

  count = 1

  gpuid = 0
  detector = RetinaFace(args.model_path, 0, gpuid, 'net3')

  # jpg 파일을 저장할 리스트를 생성합니다.
  jpg_files = []

  with open(args.input_file) as f:
    input_path = json.load(f)

  # 탐색할 디렉토리 경로 설정 (하위 폴더 포함)
  directory_path = input_path['img_dir']

  # 디렉토리 내 모든 jpg 파일 찾기
  for root, dirs, files in os.walk(directory_path):
      for file in files:
          if file.endswith(".jpg"):
              jpg_files.append(os.path.join(root, file))

  output_list = [] # 결과 JSON 데이터 저장할 리스트

  for jpg_file in tqdm(jpg_files):
    img = cv2.imread(jpg_file)
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    #im_scale = 1.0
    #if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    new_scales = [im_scale]
    flip = False

    for c in range(count):
        faces, landmarks = detector.detect(img,
                                          thresh,
                                          scales=new_scales,
                                          do_flip=flip)

    if faces is not None:
        face_data = []

        for i in range(faces.shape[0]):
            box = faces[i]
            bbox = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            confidence = float(box[4])

            face_info = {
              "bbox": bbox,
              "confidence": confidence
            }

            face_data.append(face_info)

        image_info = {
            "img_path": jpg_file,
            "faces": face_data
        }
    output_list.append(image_info)

  with open(args.output_file, 'w') as of:
    json.dump(output_list, of, indent=4)
  
if __name__ == '__main__':
  main()