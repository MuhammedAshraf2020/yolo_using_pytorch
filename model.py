import torch.nn as nn
import torch 

arch = [
		[128 , 256  , 1 ,  0] ,
		[256 , 512  , 1 ,  1] ,
		[265 , 512  , 4 ,  0] ,
		[512 , 1024 , 1 ,  1] ,
		[512 , 1024 , 2 ,  1] ]


class DarkLayer(nn.Module):
  def __init__(self , in_layer , out_layer , kernel_size  , stride , padding , bias = False):
    super(DarkLayer , self).__init__()
    self.layer = nn.Sequential(

				nn.Conv2d(in_channels = in_layer , out_channels = out_layer , 
							 kernel_size = kernel_size , stride = stride , padding = padding , bias = bias) ,
				
				nn.BatchNorm2d(out_layer) ,
				nn.LeakyReLU(0.1)    ,
			)
  def forward(self , x):
			return self.layer(x)

class Darkbox(nn.Module):
	def __init__(self  , in1  , out1 , out2):
		super(Darkbox , self).__init__()
		self.model = nn.Sequential(
				DarkLayer(in_layer = in1  , out_layer = out1 , kernel_size = 1 , stride = 1 , padding =  0 ) ,
				DarkLayer(in_layer = out1 , out_layer = out2 , kernel_size = 3 , stride = 1 , padding =  1 ) , 
			)
	def forward(self , x):
		return self.model(x)



class Darknet(nn.Module):
  def __init__(self , S = 7 , B = 2 , C = 20):
    super(Darknet , self).__init__()
    self.B = B
    self.C = C
    self.S = S
    self.initial_model      = nn.Sequential(
				DarkLayer(in_layer = 3  , out_layer = 64  , kernel_size = 7 , stride = 2 , padding = 3 ) ,
				DarkLayer(in_layer = 64 , out_layer = 192 , kernel_size = 3 , stride = 2 , padding = 1 ) ,
			)
    self.features_extractor = self.CreateBody()
    self.model_tail         = self.CreateTail()
  
  def CreateBody(self):
    layers = []
    current_channels = 192
    
    for idx,  box in enumerate(arch):	
      MaxPooling = box[3]
      out_one    = arch[idx][0]
      out_three  = arch[idx][1]
    
      for itr in range(box[2]):
        layers.append(Darkbox(current_channels , out_one , out_three))
        current_channels = out_three
    
      if MaxPooling == 1:
        layers.append(nn.MaxPool2d(kernel_size = (2 , 2) , stride = (2 , 2)))
    
    layers.append(DarkLayer(in_layer = 1024 , out_layer = 1024 , kernel_size = 3 , stride = 1 , padding = 1))
    layers.append(DarkLayer(in_layer = 1024 , out_layer = 1024 , kernel_size = 3 , stride = 2 , padding = 1))
    layers.append(DarkLayer(in_layer = 1024 , out_layer = 1024 , kernel_size = 3 , stride = 1 , padding = 1))
    layers.append(DarkLayer(in_layer = 1024 , out_layer = 1024 , kernel_size = 3 , stride = 1 , padding = 1))
    return nn.Sequential(*layers)

  def CreateTail(self):
    return nn.Sequential(
	 			nn.Linear(1024 * self.S * self.S , 496),
	 			nn.Dropout(0.3),
	 			nn.LeakyReLU(0.1),
	 			nn.Linear(496 , self.S * self.S * (5 * self.B + self.C))
	 		)
  def forward(self , x):
    x = self.initial_model(x)
    x = self.features_extractor(x)
    #print(x.shape)
    x = torch.flatten(x)
    #print(x.shape)
    return self.model_tail(x)
