from   tqdm import tqdm
import pandas as pd
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("txt")
parser.add_argument("img")

args = parser.parse_args()
train = pd.DataFrame(columns = ["image" , "class"] , index = None)
test  = pd.DataFrame(columns = ["image" , "class"] , index = None)

train_size = int(len(os.listdir(args.txt)) * 0.7 )

for index , file in tqdm(enumerate(os.listdir(args.txt)[:train_size])):
	img_path        = os.path.join(args.img , file.replace("txt" , "jpg"))
	annotate_path   = os.path.join(args.txt , file)
	train.loc[index] = [img_path , annotate_path]

train.to_csv("train.csv" , index = False) 

for index , file in tqdm(enumerate(os.listdir(args.txt)[train_size:])):
	img_path        = os.path.join(args.img , file.replace("txt" , "jpg"))
	annotate_path   = os.path.join(args.txt , file)
	test.loc[index] = [img_path , annotate_path]

test.to_csv("test.csv" , index = False)


