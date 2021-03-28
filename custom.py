import os
import torch
import pandas as pd
from PIL import Image

class VocDataSet(torch.utils.data.Dataset):
	def __init__(self , csv_file ,  S = 7 , B = 2 , C = 20 , transform = None):
		self.annotation = pd.read_csv(csv_file)
		self.transform  = transform
		self.S = S
		self.B = B
		self.C = C

	def __len__(self):
		return len(self.annotation)

	def __getitem__(self , index):
		label_path = self.annotation.iloc[index , 1]
		boxes = []
		with open(label_path) as f:
			for label in f.readlines():
				class_label , x , y , w , h = [float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n" , "").split()]
				boxes.append([class_label , x , y , w , h])

		img_path = self.annotation.iloc[index , 0]
		img = Image.open(img_path)
		boxes = torch.Tensor(boxes)

		if self.transform:
			img , boxes = self.transform(img , boxes)

		label_matrix = torch.zeros((self.S , self.S , self.C + 5 * self.B))
		for box in boxes:
			class_label , x , y , w , h = box.tolist()
			class_label = int(class_label)
			i , j = int(self.S * y) , int(self.S * x)
			x_cell , y_cell = self.S * x - j , self.S * y - i 
			width_cell , height_cell = (w * self.S , h * self.S ) # relative to the entire image
			
			if label_matrix[i , j , self.C] == 0:
				label_matrix[i , j , self.C] = 1
				box_coordintes = torch.tensor([x_cell , y_cell , width_cell , height_cell])
				label_matrix[i , j , self.C + 1:self.C + 5] = box_coordintes
				label_matrix[i , j , class_label] = 1
		return img , label_matrix