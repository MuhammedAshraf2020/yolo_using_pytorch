import torch.nn as nn
import torch

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],

]

class DarkLayer(nn.Module):
	def __init__(self , in_channels , out_channels , **kwargs):
		super(DarkLayer , self).__init__()
		self.Conv2D    = nn.Conv2d(in_channels , out_channels , bias = False , **kwargs)
		self.BatchNorm2d = nn.BatchNorm2d(out_channels)
		self.leakyrelu = nn.LeakyReLU(0.1)

	def forward(self , x):
		return self.leakyrelu(self.BatchNorm2d(self.Conv2D(x)))

class Yolov1(nn.Module):
	def __init__(self , in_channels = 3 , **kwargs):
		super(Yolov1 , self).__init__()
		self.architecture = architecture_config
		self.in_channels  = in_channels
		self.darknet      = self.create_body(self.architecture)
		self.fcs          = self.create_tail(**kwargs)

	def forward(self , x):
		return self.fcs(torch.flatten(self.darknet(x) , start_dim = 1))

	def create_body(self , architecture):
		layers = []
		in_channels = self.in_channels
		for x in architecture:
			T = type(x)
			# if its single layer then add it
			if T == tuple:
				layers += [DarkLayer(in_channels , x[1] , kernel_size = x[0] , stride = x[2] , padding = x[3])]
				in_channels = x[1]
			# if its string then its maxpooling
			elif T == str:
				layers += [nn.MaxPool2d(kernel_size = (2 , 2) , stride = (2 , 2))]
			# if its some of layers loop through every one
			elif T == list:
				conv1    = x[0]
				conv2    = x[1]
				num_iter = x[2]
				for iter in range(num_iter):
					layers += [DarkLayer(in_channels , conv1[1] , kernel_size = conv1[0] , stride = conv1[2] , padding = conv1[3])]
					layers += [DarkLayer(conv1[1] , conv2[1] , kernel_size = conv2[0] , stride = conv2[2] , padding = conv2[3])]
					in_channels = conv2[1]

		return nn.Sequential(*layers)

	def create_tail(self , S , B , C):
		return nn.Sequential(nn.Flatten() , 
							 nn.Linear(1024 * S * S , 496) ,
							 nn.Dropout(0.3) ,
							 nn.LeakyReLU(0.1) ,
							 nn.Linear(496 , S * S * ( C + B * 5)))