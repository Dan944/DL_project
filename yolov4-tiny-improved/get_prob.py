import os
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
from utils.utils import get_classes

'''
This file is aimed to generate bayes weight
'''

classes_path = 'model_data/voc_classes.txt'
class_names, _ = get_classes(classes_path)
path = "bayes/image-info"
files = os.listdir(path)
class_samples = {}
class_prob = {}
class_relation = np.zeros([20, 20], float)
total_samples = []
VOCdevkit_path = 'VOCdevkit'
bayes_path = 'bayes'
image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/trainval.txt")).read().strip().split()

if not os.path.exists(bayes_path):
    os.makedirs(bayes_path)
if not os.path.exists(os.path.join(bayes_path, 'image-info')):
    os.makedirs(os.path.join(bayes_path, 'image-info'))

for image_id in tqdm(image_ids):
    with open(os.path.join(bayes_path, "image-info/" + image_id + ".txt"), "w") as new_f:
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
            new_f.write("%s\n" % obj_name)

for label in class_names:
    class_samples[label] = 0
    class_prob[label] = 0

s = []
total_obj = 0
total_img = 0
for file in files:
    if not os.path.isdir(file):
        f = open(path + "/" + file)
        iter_f = iter(f)
        single_samples = []
        for label in iter_f:
            class_samples[label[:-1]] += 1
            total_obj += 1
            single_samples.append(class_names.index(label[:-1]))
        total_img += 1
        total_samples.append(single_samples)

for label in class_names:
    class_prob[label] = class_samples[label] / total_img
for condition in range(20):
    sub_samples = [total_samples[i] for i in range(total_img) if condition in total_samples[i]]
    for sub_sample in sub_samples:
        for label in range(20):
            if label in sub_sample:
                class_relation[condition, label] += 1
    class_relation[condition, :] = class_relation[condition, :] / len(sub_samples)

condition_prob_file = open("bayes/condition_prob.txt", "w")
for row in class_relation:
    for col in row[:-1]:
        condition_prob_file.write("%.5f," % col)
    condition_prob_file.write("%.5f\n" % row[-1])
condition_prob_file.close()

pre_prob_file = open("bayes/pre_prob.txt", "w")
for item in range(len(class_prob)-1):
    pre_prob_file.write("%.5f," % list(class_prob.values())[item])
pre_prob_file.write("%.5f\n" % list(class_prob.values())[len(class_prob)-1])
pre_prob_file.close()
