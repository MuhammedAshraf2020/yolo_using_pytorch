
import argparse
from xml.etree import ElementTree as ET
import os
from pickle import dump
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("save")
args = parser.parse_args()

path = os.path.join(args.dir)
classes_nums = {"cat" : 0 , "dog" : 1}

keys = list(classes_nums.keys())
try:
	os.mkdir(args.save)
except:
	print("Folder is already exist !")

def ToMidPoint(x1 , y1 , x2 , y2 , size):
  dw = 1.0 / size[0]
  dh = 1.0 / size[1]
  h  = y2 - y1 
  w  = x2 - x1
  x  = (x1 + (w/2)) 
  y  = (y1 + (h/2))
  return x * dw , y * dh , w * dw , h * dh

for File in tqdm(os.listdir(path)):
  obj_list = 0
  xml_path  = os.path.join(path , File)
  file_name = "{}/{}".format(args.save , File.replace("xml" , "txt"))
  tree      = ET.parse(xml_path)
  root      = tree.getroot()
  size = root.find('size')
  w_img = int(size.find('width').text)
  h_img = int(size.find('height').text)
  with open(file_name , "w") as F :
    for obj in root.iter("object"):
      class_name = obj.find("name").text
      if class_name not in keys:
        continue
      obj_list += 1
      class_id      = classes_nums[class_name]
      xml_box       = obj.find("bndbox")
      nedded        = ["xmin" , "ymin" , "xmax" , "ymax"]
      x1 , y1       = float(xml_box.find(nedded[0]).text) , float(xml_box.find(nedded[1]).text)
      x2 , y2       = float(xml_box.find(nedded[2]).text) , float(xml_box.find(nedded[3]).text)
      x , y , w , h = ToMidPoint(x1 , y1 , x2 , y2 , (w_img , h_img)) 
      F.write("{} {} {} {} {}\n".format(class_id , x , y , w , h))
    if obj_list == 0:
      os.remove(file_name)
