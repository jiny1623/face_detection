import argparse

import cv2
import numpy as np
import torch
import json
import os

from pathlib import Path
from tqdm.auto import tqdm

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    parser.add_argument('config', help='config file path')
    # parser.add_argument('imgname', help='image file name')
    parser.add_argument('json', help='input json file name')
    args = parser.parse_args()
    return args


def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def plot_result(result, imgfp, class_names, outfp='out.jpg'):
    font_scale = 0.5
    bbox_color = 'green'
    text_color = 'green'
    thickness = 1

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = imread(imgfp)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    imwrite(img, outfp)


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    json_fn = args.json
    with open(json_fn) as f:
      input_path = json.load(f)

    print("Input Image Directory: ", input_path['img_dir'])

    jpg_files = []

    directory_path = input_path['img_dir']

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".jpg"):
                jpg_files.append(os.path.join(root, file))

    class_names = cfg.class_names

    engine, data_pipeline, device = prepare(cfg)
    
    # data = []

    # for jpg_file in jpg_files:
    #   data.append(dict(img_info=dict(filename=jpg_file), img_prefix=None))
    for jpg_file in tqdm(jpg_files):
      data = dict(img_info=dict(filename=jpg_file), img_prefix=None)

      data = data_pipeline(data)
      data = collate([data], samples_per_gpu=1)

      if device != 'cpu':
          # scatter to specified GPU
          data = scatter(data, [device])[0]
      else:
          # just get the actual data from DataContainer
          data['img_metas'] = data['img_metas'][0].data
          data['img'] = data['img'][0].data

      result = engine.infer(data['img'], data['img_metas'])[0]

      bboxes = np.vstack(result)
      labels = [
          np.full(bbox.shape[0], idx, dtype=np.int32)
          for idx, bbox in enumerate(result)
      ]
      labels = np.concatenate(labels)

      # write txt

      new_root = '/content/drive/MyDrive/rippleai/submission/tinaface_v1/images/'
      rel_path = os.path.relpath(jpg_file, '/content/drive/MyDrive/rippleai/WIDER_val/images/')
      new_path = Path(os.path.join(new_root, rel_path)).with_suffix('.txt')
      createDirectory(os.path.dirname(new_path))

      of = open(new_path, "w+")
      of.write(os.path.splitext(os.path.basename(new_path))[0] + "\n")
      of.write(str(len(bboxes)) + "\n")

      for bbox, label in zip(bboxes, labels):
          x = float(bbox[0])
          y = float(bbox[1])
          w = float(bbox[2]) - float(bbox[0])
          h = float(bbox[3]) - float(bbox[1])
          confidence = float(bbox[4])
          inf = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + str(confidence) + "\n"
          of.write(inf)
      of.close()

if __name__ == '__main__':
    main()
