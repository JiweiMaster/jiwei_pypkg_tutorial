import multiprocessing
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import xml.etree.ElementTree
from PIL import Image
from pascal_voc_writer import Writer
import json
from pathlib import Path
from PIL import Image


class Config:
    def __init__(self, image_dir, label_dir, names):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.names = names

def yolo2voc(txt_file, config:Config):
    w, h = Image.open(os.path.join(config.image_dir, f'{txt_file[:-4]}.jpg')).size
    writer = Writer(f'{txt_file[:-4]}.xml', w, h)
    with open(os.path.join(config.label_dir, txt_file)) as f:
        for line in f.readlines():
            label, x_center, y_center, width, height = line.rstrip().split(' ')
            x_min = int(w * max(float(x_center) - float(width) / 2, 0))
            x_max = int(w * min(float(x_center) + float(width) / 2, 1))
            y_min = int(h * max(float(y_center) - float(height) / 2, 0))
            y_max = int(h * min(float(y_center) + float(height) / 2, 1))
            writer.addObject(config.names[int(label)], x_min, y_min, x_max, y_max)
    print(os.path.join(config.label_dir, f'{txt_file[:-4]}.xml'))
    writer.save(os.path.join(config.label_dir, f'{txt_file[:-4]}.xml'))

def voc2yolo(xml_file, config:Config):
    in_file = open(f'{config.label_dir}/{xml_file}')
    try:
        root = xml.etree.ElementTree.parse(in_file).getroot()
    except:
        return
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    has_class = False
    for obj in root.iter('object'):
        name = obj.find('name').text
        if name in config.names:
            has_class = True
    if has_class:
        print(f'{config.label_dir}/{xml_file[:-4]}.txt')
        out_file = open(f'{config.label_dir}/{xml_file[:-4]}.txt', 'w')
        for obj in root.iter('object'):
            name = obj.find('name').text
            if name in config.names:
                xml_box = obj.find('bndbox')
                x_min = float(xml_box.find('xmin').text)
                y_min = float(xml_box.find('ymin').text)
                x_max = float(xml_box.find('xmax').text)
                y_max = float(xml_box.find('ymax').text)

                box_x = (x_min + x_max) / 2.0 - 1
                box_y = (y_min + y_max) / 2.0 - 1
                box_w = x_max - x_min
                box_h = y_max - y_min
                box_x = box_x * 1. / w
                box_w = box_w * 1. / w
                box_y = box_y * 1. / h
                box_h = box_h * 1. / h

                b = [box_x, box_y, box_w, box_h]
                cls_id = config.names.index(obj.find('name').text)
                out_file.write(str(cls_id) + " " + " ".join([str(f'{a:.6f}') for a in b]) + '\n')

def voc_yolo_convert(config:Config, f_yolo2voc = False, f_voc2yolo = False):
    '''
    voc和yolo数据格式的转化， 其中f_yolo2voc,f_voc2yolo一个是True，另外一个是False才行
    # image_dir, label_dir, names
    image_dir = "image_test/images"
    label_dir = "demo_outputs"
    names= ['door','cabinetDoor','refrigeratorDoor','window']
    '''
    if f_yolo2voc and not f_voc2yolo:
        print('YOLO to VOC')
        txt_files = [name for name in os.listdir(config.label_dir) if name.endswith('.txt')]
        with multiprocessing.Pool(os.cpu_count()) as pool:
            pool.starmap(yolo2voc, txt_files, list(zip(xml_files, [config for i in xml_files])))
        pool.close()
        pool.join()

    if f_voc2yolo and not f_yolo2voc:
        print('VOC to YOLO')
        xml_files = [name for name in os.listdir(config.label_dir) if name.endswith('.xml')]
        print("Waiting transfer files: ", len(xml_files))
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            pool.starmap(voc2yolo, list(zip(xml_files, [config for i in xml_files])))
        pool.close()
        pool.join()


class YOLO2COCO:
    def __init__(self, root, split, COCO_CATEGORIES):
        self.info = {
            "year": 2021,
            "version": "1.0",
            "description": "For object detection",
            "date_created": "2021",
        }
        self.licenses = [
            {
                "id": 1,
                "name": "GNU General Public License v3.0",
                "url": "https://github.com/zhiqwang/yolov5-rt-stack/blob/master/LICENSE",
            }
        ]
        self.type = "instances"
        self.split = split
        self.root_path = Path(root)
        self.label_path = self.root_path / "labels"
        self.annotation_root = self.root_path / "annotations"
        Path(self.annotation_root).mkdir(parents=True, exist_ok=True)

        self.categories = [
            {
                "id": coco_category["id"],
                "name": coco_category["name"],
                "supercategory": coco_category["supercategory"],
            }
            for coco_category in COCO_CATEGORIES
        ]

    def generate(self, coco_type="instances", annotation_format="bbox"):
        print(self.label_path)
        label_paths = sorted(self.label_path.rglob("*.txt"))
        images, annotations = self._get_image_annotation_pairs(
            label_paths,
            annotation_format=annotation_format,
        )
        json_data = {
            "info": self.info,
            "images": images,
            "licenses": self.licenses,
            "type": self.type,
            "annotations": annotations,
            "categories": self.categories,
        }
        output_path = self.annotation_root / f"{coco_type}_{self.split}.json"
        with open(output_path, "w") as json_file:
            json.dump(json_data, json_file, sort_keys=True)
    
    def isExist(self, file_path):
        if os.path.exists(file_path):
            return True
        else:
            False

    def _get_image_annotation_pairs(self, label_paths, annotation_format="bbox"):
        images = []
        annotations = []
        annotation_id = 0
        for img_id, label_path in enumerate(label_paths, 1):
            img_path = str(label_path).replace("labels", "images").replace(".txt", ".jpg")
            img = Image.open(img_path)
            width, height = img.size

            images.append(
                {
                    "date_captured": "2021",
                    "file_name": str(Path(img_path).relative_to(self.root_path)),
                    "id": img_id,
                    "license": 1,
                    "url": "",
                    "height": height,
                    "width": width,
                }
            )

            with open(label_path, "r") as f:
                for line in f:
                    label_info = line.strip().split()
                    assert len(label_info) == 5
                    annotation_id += 1

                    category_id, vertex_info = label_info[0], label_info[1:]
                    category_id = self.categories[int(category_id)]["id"]
                    if annotation_format == "bbox":
                        segmentation, bbox, area = self._get_annotation(
                            vertex_info, height, width
                        )
                    else:
                        raise NotImplementedError

                    annotations.append(
                        {
                            "segmentation": segmentation,
                            "area": area,
                            "iscrowd": 0,
                            "image_id": img_id,
                            "bbox": bbox,
                            "category_id": category_id,
                            "id": annotation_id,
                        }
                    )

        return images, annotations

    @staticmethod
    def _get_annotation(vertex_info, height, width):

        cx, cy, w, h = [float(i) for i in vertex_info]
        cx = cx * width
        cy = cy * height
        w = w * width
        h = h * height
        x = cx - w / 2
        y = cy - h / 2

        segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        area = w * h

        bbox = [x, y, w, h]
        return segmentation, bbox, area

def yolo_coco_convert(data_path, split="train", COCO_CATEGORIES=[]):
    '''
    yolo dataset to coco dataset
    COCO_CATEGORIES的格式需要指定
    COCO_CATEGORIES = [
    {
        "id": 0,
        "color": [220, 20, 60],
        "isthing": 1,
        "name": "door",
        "supercategory": "",
    }]
    '''
    converter = YOLO2COCO(data_path, split, COCO_CATEGORIES)
    converter.generate()

class COCO2YOLO:
    def __init__(self, json_file, output):
        self.json_file = json_file
        self.output = output
        self._check_file_and_dir(json_file, output)
        self.labels = json.load(open(json_file, 'r', encoding='utf-8'))
        self.coco_id_name_map = self._categories()
        self.coco_name_list = list(self.coco_id_name_map.values())
        print("total images", len(self.labels['images']))
        print("total categories", len(self.labels['categories']))
        print("total labels", len(self.labels['annotations']))

    def _check_file_and_dir(self, file_path, dir_path):
        if not os.path.exists(file_path):
            raise ValueError("file not found")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _categories(self):
        categories = {}
        for cls in self.labels['categories']:
            categories[cls['id']] = cls['name']
        return categories

    def _load_images_info(self):
        images_info = {}
        for image in self.labels['images']:
            id = image['id']
            file_name = image['file_name']
            if file_name.find('\\') > -1:
                file_name = file_name[file_name.index('\\')+1:]
            w = image['width']
            h = image['height']
            images_info[id] = (file_name, w, h)

        return images_info

    def _bbox_2_yolo(self, bbox, img_w, img_h):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        centerx = bbox[0] + w / 2
        centery = bbox[1] + h / 2
        dw = 1 / img_w
        dh = 1 / img_h
        centerx *= dw
        w *= dw
        centery *= dh
        h *= dh
        return centerx, centery, w, h

    def _convert_anno(self, images_info):
        anno_dict = dict()
        for anno in self.labels['annotations']:
            bbox = anno['bbox']
            image_id = anno['image_id']
            category_id = anno['category_id']

            image_info = images_info.get(image_id)
            image_name = image_info[0]
            img_w = image_info[1]
            img_h = image_info[2]
            yolo_box = self._bbox_2_yolo(bbox, img_w, img_h)

            anno_info = (image_name, category_id, yolo_box)
            anno_infos = anno_dict.get(image_id)
            if not anno_infos:
                anno_dict[image_id] = [anno_info]
            else:
                anno_infos.append(anno_info)
                anno_dict[image_id] = anno_infos
        return anno_dict

    def save_classes(self):
        sorted_classes = list(map(lambda x: x['name'], sorted(self.labels['categories'], key=lambda x: x['id'])))
        print('coco names', sorted_classes)
        with open('coco.names', 'w', encoding='utf-8') as f:
            for cls in sorted_classes:
                f.write(cls + '\n')
        f.close()

    def coco2yolo(self):
        print("loading image info...")
        images_info = self._load_images_info()
        print("loading done, total images", len(images_info))

        print("start converting...")
        anno_dict = self._convert_anno(images_info)
        print("converting done, total labels", len(anno_dict))

        print("saving txt file...")
        self._save_txt(anno_dict)
        print("saving done")

    def _save_txt(self, anno_dict):
        for k, v in anno_dict.items():
            file_name = os.path.splitext(v[0][0])[0] + ".txt"
            with open(os.path.join(self.output, file_name), 'w', encoding='utf-8') as f:
                print(k, v)
                for obj in v:
                    cat_name = self.coco_id_name_map.get(obj[1])
                    category_id = self.coco_name_list.index(cat_name)
                    box = ['{:.6f}'.format(x) for x in obj[2]]
                    box = ' '.join(box)
                    line = str(category_id) + ' ' + box
                    f.write(line + '\n')

def coco_yolo_convert(json_file, output):
    '''
    coco dataset convert to yolo dataset
    '''
    c2y = COCO2YOLO(json_file, output)
    c2y.coco2yolo()

def map_score(gt_path, pred_path):
    '''
    yolo系列预测的数据和gt来计算map
    '''
    annoFile = gt_path
    coco = COCO(annoFile)
    pred_annotations = json.load(open(pred_path,"r"))
    
    # 初始化 COCOeval 对象
    cocoGt = coco
    cocoDt = cocoGt.loadRes(pred_annotations)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    # 运行评估
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    # ap50
    ap50 = int(cocoEval.stats[1]*10000)
    return ap50

