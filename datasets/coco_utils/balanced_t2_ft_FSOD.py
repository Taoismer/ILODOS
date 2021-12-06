import itertools
import random
import os
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager

from detectron2.utils.store_non_list import Store


VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

T2_cate_file = './ImageSets/t2/t2_train_cate_fsod.txt'
with open(T2_cate_file,'r') as f:
    cate_list = f.readlines()
    T2_CLASS_NAMES = [c.strip() for c in cate_list]
    print('number of class in T2', len(set(T2_CLASS_NAMES)))

# T3_cate_file = './ImageSets/t3/t3_train_cate_fsod.txt'
# with open(T3_cate_file,'r') as f:
#     cate_list = f.readlines()
#     T3_CLASS_NAMES = [c.strip() for c in cate_list]
#     print('number of class in T3', len(set(T3_CLASS_NAMES)))

# T4_cate_file = './ImageSets/t4/t4_train_cate_fsod.txt'
# with open(T4_cate_file,'r') as f:
#     cate_list = f.readlines()
#     T4_CLASS_NAMES = [c.strip() for c in cate_list]
#     print('number of class in T4', len(set(T4_CLASS_NAMES)))

UNK_CLASS = ["unknown"]

# Change this accodingly for each task t*
known_classes = VOC_CLASS_NAMES + T2_CLASS_NAMES 
train_files = ['./ImageSets/t2/t2_train_fsod.txt','../VOC2007/ImageSets/Main/t1_train.txt']

# known_classes = list(itertools.chain(VOC_CLASS_NAMES))
annotation_location = '/home/causal_ws/OWOD/datasets/VOC2007/Annotations'

items_per_class = 1
dest_file = './ImageSets/t2/t2_ft_fsod_' + str(items_per_class) + '.txt'

file_names = []
for tf in train_files:
    with open(tf, mode="r") as myFile:
        file_names.extend(myFile.readlines())

random.shuffle(file_names)

image_store = Store(len(known_classes), items_per_class)

current_min_item_count = 0

_classes = []

for fileid in file_names:
    fileid = fileid.strip()
    anno_file = os.path.join(annotation_location, fileid + ".xml")

    with PathManager.open(anno_file) as f:
        tree = ET.parse(f)

    for obj in tree.findall("object"):
        cls = obj.find("name").text
        _classes.append(cls) #to assert
        if cls in known_classes:
            image_store.add((fileid,), (known_classes.index(cls),))


    current_min_item_count = min([len(items) for items in image_store.retrieve(-1)])
    # print(current_min_item_count)
    if current_min_item_count == items_per_class:
        break

for c in known_classes:
    assert c in _classes, '{} class do not find instance '.format(c)


filtered_file_names = []
for items in image_store.retrieve(-1):
    filtered_file_names.extend(items)

print(image_store)
print(len(filtered_file_names))
print(len(set(filtered_file_names)))

filtered_file_names = set(filtered_file_names)
filtered_file_names = map(lambda x: x + '\n', filtered_file_names)

with open(dest_file, mode="w") as myFile:
    myFile.writelines(filtered_file_names)

print('Saved to file: ' + dest_file)
