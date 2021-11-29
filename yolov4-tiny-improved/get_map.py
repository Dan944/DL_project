import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from yolo import YOLO
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map

'''
This file is aimed to test the mAP of the model
'''

if __name__ == "__main__":
    # classes_path, the path to VOC class names
    classes_path = 'model_data/voc_classes.txt'
    #   MINOVERLAP  threshold to map, like map0.5 is here
    MINOVERLAP = 0.5
    #   map_vis to generate visible result
    map_vis = False
    #   VOC data path
    VOCdevkit_path = 'VOCdevkit'
    # map_out, the output path
    map_out_path = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()
    # image_ids = image_ids[:946]

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    print("Load model.")
    yolo = YOLO(confidence=0.001, nms_iou=0.5)
    print("Load model done.")

    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
        image = Image.open(image_path)
        if map_vis:
            image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
        yolo.get_map_txt(image_id, image, class_names, map_out_path)
    print("Get predict result done.")

    print("Get ground truth result.")
    for image_id in tqdm(image_ids):
        with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
            root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/" + image_id + ".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult') != None:
                    difficult = obj.find('difficult').text
                    if int(difficult) == 1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Get ground truth result done.")

    print("Get map.")
    get_map(MINOVERLAP, True, path=map_out_path)
    print("Get map done.")

    print("Get map.")
    get_coco_map(class_names=class_names, path=map_out_path)
    print("Get map done.")
