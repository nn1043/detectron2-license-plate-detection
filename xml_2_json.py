from bs4 import BeautifulSoup
import lxml
import json
import base64
import os

filenames = []
widths = []
heights = []
shapes = {}

path_anno = "/home/nicholas/Downloads/annotations/"
for annotation in os.listdir(path_anno):
    file = open(path_anno + annotation, "r")
    xml = BeautifulSoup(file.read(), "lxml")
    xml = xml.find("annotation")
    file.close()

    filename = str(xml.find("filename").text)
    filenames.append(filename)
    widths.append( int(xml.find("size").find("width").text) )
    heights.append( int(xml.find("size").find("height").text) )
    license_plates = xml.find_all("object")

    shapes[filename] = []
    for license_plate in license_plates:
        bndbox = license_plate.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        bbox = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]
        shapes[filename].append( { "label":"LICENSE_PLATE", "points":bbox, "group_id":None, "shape_type":"polygon", "flags":{} } )

image_binary = []
path_images = "/home/nicholas/Downloads/images/"
for i in range(len(filenames)):
    with open(path_images + filenames[i], "rb") as image_file:
        image_binary.append( base64.b64encode(image_file.read()).decode("ascii") )


for i in range(len(filenames)):
    label = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes[filenames[i]],
        "imagePath": filenames[i],
        "imageData": image_binary[i],
        "imageHeight": heights[i],
        "imageWidth": widths[i],
    }

    path_output = "/home/nicholas/Downloads/labels"
    json_string = json.dumps(label)
    with open(f"{path_output}/{filenames[i][:-4]}.json", "w") as output:
        output.write(json_string)
