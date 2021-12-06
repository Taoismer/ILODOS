import json
from numpy.lib.function_base import select
import pandas

json_path = '/home/causal_ws/Datasets/FSOD/fsod/annotations/fsod_train.json'

with open(json_path, 'r') as f:
    data = json.load(f)

print(data.keys())

ALL_CLASS_NAMES = []
IDS= []

# print(data['categories'])

VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

_pre_id = -1 # to assert
for c in data['categories']:
    cate_name = c['name']
    cate_id = c['id']
    ALL_CLASS_NAMES.append(cate_name)
    if cate_name not in VOC_CLASS_NAMES:
        IDS.append(cate_id)
    assert _pre_id != cate_id, 'error ID label {}:{}'.format(cate_name,cate_id)
    _pre_id = cate_id


IDS = IDS[600:] # start from id 1
print(max(IDS))

Imgs = []
Annos = []
instance_per_cls = 1

from tqdm import tqdm
for i in tqdm(IDS):
    assert ALL_CLASS_NAMES[i-1] not in VOC_CLASS_NAMES, 'FSOD Dataset include same class as VOC, id is: {}'.format(i)
    anno_frame = pandas.DataFrame(data['annotations'])
    #find some images containing the category 
    sample_instance = anno_frame[anno_frame['category_id']==i] #.sample(instance_per_cls)
    sample_img_id = sample_instance['image_id'].tolist()
    # print(sample_img_id, ALL_CLASS_NAMES[i-1])
    #find all other instance annnotation in the images of same class
    all_annos = anno_frame[anno_frame['image_id'].isin(sample_img_id)]
    for j in sample_img_id:
        annos_per_img = all_annos[all_annos['image_id']==j]
        annos_per_img = annos_per_img.to_dict('index')
        annos_per_img = list(annos_per_img.values())
        Annos.append(annos_per_img)

    imgs_frame = pandas.DataFrame(data['images'])
    # find img path corresponding to the image id
    for j in sample_img_id:
        sample_imgs = imgs_frame[imgs_frame['id']==j]
        sample_imgs = sample_imgs.to_dict('index')
        sample_imgs = list(sample_imgs.values())
        Imgs.append(sample_imgs)


Imgs_and_Annos = []
assert len(Annos)==len(Imgs), 'length of annoations and images is not equal'
for i in range(len(Imgs)):
    Imgs_and_Annos.append([Imgs[i]]+[Annos[i]])

# from pprint import pprint
# print('>>>>>>>> Number of selected images:', Imgs_and_Annos)

import xml.etree.cElementTree as ET
import os
from shutil import copyfile
source_folder = '/home/causal_ws/Datasets/FSOD/'
target_folder = '/home/causal_ws/OWOD/datasets/coco_utils/ImageSets/t4'
if not os.path.isdir(target_folder):
    os.mkdir(target_folder)

selected_ids = []
selected_categories = []

for ia in tqdm(Imgs_and_Annos):
    selected_ids.append(str(ia[0][0]['id']))

    # img_name = str(ia[0][0]['id'])+'.jpg'
    # copyfile(os.path.join(source_folder, ia[0][0]['file_name']), os.path.join(target_folder, 'Images', img_name))
    # annotation_el = ET.Element('annotation')
    # ET.SubElement(annotation_el, 'filename').text = img_name
    # size_el = ET.SubElement(annotation_el, 'size')
    # ET.SubElement(size_el, 'width').text = str(ia[0][0]['width'])
    # ET.SubElement(size_el, 'height').text = str(ia[0][0]['height'])
    # ET.SubElement(size_el, 'depth').text = str(3)
    
    for _a in ia[1:]:# images of same class
        for a in _a:
            selected_categories.append(ALL_CLASS_NAMES[a['category_id']-1])
    #         
    #         object_el = ET.SubElement(annotation_el, 'object')
    #         ET.SubElement(object_el,'name').text = ALL_CLASS_NAMES[a['category_id']-1]
    #         # print('category:',ALL_CLASS_NAMES[a['category_id']], 'id:', a['category_id'], '\n')
    #         # ET.SubElement(object_el, 'name').text = 'unknown'
    #         ET.SubElement(object_el, 'difficult').text = '0'
    #         bb_el = ET.SubElement(object_el, 'bndbox')
    #         ET.SubElement(bb_el, 'xmin').text = str(int(a['bbox'][0]))
    #         ET.SubElement(bb_el, 'ymin').text = str(int(a['bbox'][1]))
    #         ET.SubElement(bb_el, 'xmax').text = str(int(a['bbox'][0] + a['bbox'][2]))
    #         ET.SubElement(bb_el, 'ymax').text = str(int(a['bbox'][1] + a['bbox'][3]))
    # ET.ElementTree(annotation_el).write(os.path.join(target_folder, 'Annotations', img_name.split('.')[0] + '.xml'))

for i in IDS:
    assert ALL_CLASS_NAMES[i-1] in selected_categories


import numpy as np
selected_ids = set(selected_ids)
id_txt = os.path.join(target_folder,'t4_test_fsod.txt')
cate_txt = os.path.join(target_folder,'t4_test_cate_fsod.txt')

sepa = '\n'
with open(id_txt, 'w') as f:
    f.write(sepa.join(selected_ids))

sepa = '\n'
_cate = []
for i in IDS:
    _cate.append(ALL_CLASS_NAMES[i-1])
with open(cate_txt, 'w') as f:
    f.write(sepa.join(_cate))

print('Done')