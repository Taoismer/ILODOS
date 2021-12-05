import json
from numpy.lib.function_base import select
import pandas
import os


json_path = '/home/causal_ws/Datasets/FSOD/fsod/annotations/fsod_test.json'

with open(json_path, 'r') as f:
    data = json.load(f)

print(data.keys())

ALL_CLASS_NAMES = []
IDS= []

# print(data['categories'])

for c in data['categories']:
    ALL_CLASS_NAMES.append(c['name'])
    IDS.append(c['id'])

csv_path = './ImageSets/FSOD'
if not os.path.isdir(csv_path):
    os.mkdir(csv_path)
cate_frame = pandas.DataFrame(ALL_CLASS_NAMES)
cate_frame.to_csv(os.path.join(csv_path, 'test_category.csv'))

# IDS = IDS[0:300]

Imgs = []
Annos = []
# instance_per_cls = 1

from tqdm import tqdm
for i in tqdm(IDS):
    anno_frame = pandas.DataFrame(data['annotations'])
    #find image containing the category 
    sample_instance = anno_frame[anno_frame['category_id']==i] #.sample(instance_per_cls)
    sample_img_id = sample_instance['image_id'].tolist()
    # print(sample_img_id, ALL_CLASS_NAMES[i-1])
    #find all other instance annnotation in the same image
    all_annos = anno_frame[anno_frame['image_id'].isin(sample_img_id)]
    all_annos = all_annos.to_dict('index')
    Annos.append(all_annos)

    imgs_frame = pandas.DataFrame(data['images'])
    # find img path corresponding to the image id
    sample_img = imgs_frame[imgs_frame['id'].isin(sample_img_id)]
    Imgs.append(sample_img.to_dict('index'))

_Annos = []
for i in Annos:
    _Annos.append(list(i.values()))
_Imgs = []
for i in Imgs:
    _Imgs.append(list(i.values()))
Annos = _Annos
Imgs = _Imgs

del _Imgs, _Annos

Imgs_and_Annos = []
assert len(Annos)==len(Imgs), 'length of annoations and images is not equal'
for c in range(len(Imgs)):
    for i in range(len(Imgs[c])):
        Imgs_and_Annos.append([Imgs[c][i]]+[Annos[c][i]])

print('>>>>>>>> Number of images:', len(Imgs_and_Annos))

import xml.etree.cElementTree as ET
from shutil import copyfile
source_folder = '/home/causal_ws/Datasets/FSOD/'
target_folder = '/home/causal_ws/OWOD/datasets/coco_utils/ImageSets/FSOD/test'
if not os.path.isdir(target_folder):
    os.mkdir(target_folder)

id_txt = os.path.join(target_folder, 'test.txt')
selected_ids = []

for ia in tqdm(Imgs_and_Annos):
    img_name = str(ia[0]['id'])+'.jpg'

    if os.path.exists(img_name):
        continue
    
    selected_ids.append(str(ia[0]['id']))
    copyfile(os.path.join(source_folder, ia[0]['file_name']), os.path.join(target_folder, 'Images', img_name))
    annotation_el = ET.Element('annotation')
    ET.SubElement(annotation_el, 'filename').text = img_name
    size_el = ET.SubElement(annotation_el, 'size')
    ET.SubElement(size_el, 'width').text = str(ia[0]['width'])
    ET.SubElement(size_el, 'height').text = str(ia[0]['height'])
    ET.SubElement(size_el, 'depth').text = str(3)
    
    for a in ia[1:]:
        object_el = ET.SubElement(annotation_el, 'object')
        ET.SubElement(object_el,'name').text = ALL_CLASS_NAMES[a['category_id']-1]
        # print('category:',ALL_CLASS_NAMES[a['category_id']], 'id:', a['category_id'], '\n')
        # ET.SubElement(object_el, 'name').text = 'unknown'
        ET.SubElement(object_el, 'difficult').text = '0'
        bb_el = ET.SubElement(object_el, 'bndbox')
        ET.SubElement(bb_el, 'xmin').text = str(int(a['bbox'][0]))
        ET.SubElement(bb_el, 'ymin').text = str(int(a['bbox'][1]))
        ET.SubElement(bb_el, 'xmax').text = str(int(a['bbox'][0] + a['bbox'][2]))
        ET.SubElement(bb_el, 'ymax').text = str(int(a['bbox'][1] + a['bbox'][3]))
    ET.ElementTree(annotation_el).write(os.path.join(target_folder, 'Annotations', img_name.split('.')[0] + '.xml'))

nxt = '\n'
with open(id_txt, 'w') as f:
    f.write(nxt.join(selected_ids))

print('Done')