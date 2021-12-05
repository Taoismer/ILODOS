from pycocotools.coco import COCO
import numpy as np


T2_CLASS_NAMES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

# Train
coco_annotation_file = './coco_annotations/instances_train2017.json'
dest_file = './ImageSets/t2_train.txt'

img_dir = '../VOC2007/JPEGImages/'

coco_instance = COCO(coco_annotation_file)

image_ids = []
select_image_ids = []
cls = []
_cls = []

for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
    if not set(classes).isdisjoint(T2_CLASS_NAMES):
        image_ids.append(image_details['file_name'].split('.')[0])
        cls.extend(classes)
        _cls.append(classes)

import random
import os
from  shutil import copy
items_per_class = 1
for C in T2_CLASS_NAMES:
    for i,img in enumerate(_cls):
        if C in img:
            select_image_ids.append(image_ids[i])
            break


(unique, counts) = np.unique(cls, return_counts=True)
# print({x:y for x,y in zip(unique, counts)})

with open(dest_file, 'w') as file:
    for image_id in select_image_ids:
        file.write(str(image_id)+'\n')
        copy(os.path.join(img_dir,str(image_id)+'.jpg'), './ImageSets/t2_train')

print('Created train file')


# Test
coco_annotation_file = './coco_annotations/instances_val2017.json'
dest_file = './ImageSets/t2_test.txt'

coco_instance = COCO(coco_annotation_file)

image_ids = []
cls = []
for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
    if not set(classes).isdisjoint(T2_CLASS_NAMES):
        image_ids.append(image_details['file_name'].split('.')[0])
        cls.extend(classes)

(unique, counts) = np.unique(cls, return_counts=True)
print({x:y for x,y in zip(unique, counts)})

with open(dest_file, 'w') as file:
    for image_id in image_ids:
        file.write(str(image_id)+'\n')
print('Created test file')

dest_file = './ImageSets/t2_test_unk.txt'
with open(dest_file, 'w') as file:
    for image_id in image_ids:
        file.write(str(image_id)+'\n')

print('Created test_unk file')
