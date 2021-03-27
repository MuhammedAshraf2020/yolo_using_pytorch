
import torch
from PIL import Image
import pandas as pd

class CustomData(torch.utils.data.Dataset):
	def __init__(self , annotate_file , transform = None):
		self.annotate_file = pd.read_csv(annotate_file)
		self.transform = transform

	def __len__(self):
		return len(self.annotate_file)

	def __getitem__(self , index):
		label = int(self.annotate_file.iloc[index , 1])
		img = Image.open(self.annotate_file.iloc[index , 0])
		if self.transform:
			img = self.transform(img)

		return img , label