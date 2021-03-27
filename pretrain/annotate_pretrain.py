
import argparse
import pandas as pd
from tqdm import tqdm
import os


parser = argparse.ArgumentParser()
parser.add_argument("train")
args   = parser.parse_args()

train  = pd.DataFrame(columns = ["image" , "class"] , index = None)
test   = pd.DataFrame(columns = ["image" , "class"] , index = None)

img_path   = os.path.join(args.train)
print(len(os.listdir(img_path)))
train_size = int(len(os.listdir(img_path)) * 0.7 )

for index , File in tqdm(enumerate(os.listdir(img_path)[:train_size])):
	img = os.path.join(img_path , File)
	if File[:3] == "cat":
		train.loc[index] = [img  , 0]
	else:
		train.loc[index] = [img , 1]

train.to_csv("train.csv" , index = False)

for index , File in tqdm(enumerate(os.listdir(img_path)[train_size:])):
	img = os.path.join(img_path , File)
	if File[:3] == "cat":
		test.loc[index] = [img , 0]
	else:
		test.loc[index]  = [img , 1]

test.to_csv("test.csv" , index = False)
