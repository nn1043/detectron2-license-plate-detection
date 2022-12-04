import torch, torchvision
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

from bs4 import BeautifulSoup
import lxml
import os
import cv2

register_coco_instances("license", {}, "labels/labels_all.json", "images/train")
license_metadata = MetadataCatalog.get("license")
dataset_dicts_train = DatasetCatalog.get("license")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "models/res_101.pth" # res_50.pth, res_101.pth
cfg.DATASETS.TRAIN = ("license")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)


ground_truth = {}
path_anno = "/home/nicholas/Downloads/annotations/"
for annotation in os.listdir(path_anno):
    file = open(path_anno + annotation, "r")
    xml = BeautifulSoup(file.read(), "lxml")
    xml = xml.find("annotation")
    file.close()

    filename = str(xml.find("filename").text)
    ground_truth[filename] = []

    license_plates = xml.find_all("object")
    for license_plate in license_plates:
        bndbox = license_plate.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        ground_truth[filename].append([center_x, center_y])

predictions = {}
total_predictions = 0
counter = 0
path_test = "/home/nicholas/dt2/images/test/"
for image in os.listdir(path_test):
    print(f"Currently evaluating: {image}")
    image_path = path_test + image
    im = cv2.imread(image_path)

    results = predictor(im)
    results = results['instances']

    pred = []
    for box in results.pred_boxes:
        center = [float((box[0]+box[2])/2), float((box[1]+box[3])/2)]
        pred.append(center)
        total_predictions += 1
    predictions[image] = pred

    counter += 1
    if counter%10 == 0:
        print(f"Iteration: {counter}")

total_accurate = 0
recall_counter = 0
for image in predictions:
    for pred in predictions[image]:
        for gt in ground_truth[image]:
            if ((abs(pred[0] - gt[0]) <= 20) and (abs(pred[1] - gt[1])) <= 20):
                total_accurate += 1
    recall_counter += len(ground_truth[image])

precision = total_accurate / total_predictions
recall = total_accurate / recall_counter
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Total Predictions: {total_predictions}")
print(f"Total Accurate: {total_accurate}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
