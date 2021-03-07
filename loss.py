import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.module):
	def __init__(self , S = 7 , B = 2 , C = 20):
		super(YoloLoss , self).__init__()
		# S * S number of the celles 7 * 7
		# B number of Bounding box 1 
		# C number of classes 20
		self.S   = S
		self.B   = B
		self.C   = C
		# mean square error function
		self.mse = nn.MSELOSS(reduction = "sum")
		# obj , nobj
		self.lambda_noobj  = 0.5
		self.lambda_coord  = 5

	def forward(self , predictions , target):

		predictions = predictions.reshape(-1 , self.S , self.S , self.C , self.B * 5)  # N , S , S , 30
		iou_b1 = intersection_over_union(predictions[... , 21:25] , target[... , 21:25]) # num_examples , iou 
		iou_b2 = intersection_over_union(predictions[... , 26:30] , target[... , 21:25]) # num_examples , iou
		ious   = torch.cat([iou_b1.unsqueeze(0) , iou_b2.unsqueeze(0)] , dim = 0) #
		iou_maxes , best_box = torch.max(ious , dim = 0)
		exists_box  = target[... , 20].unsqueeze(3)
		
		# loss box cords
		box_predictions = exists_box * ( # ( examples , 4 )
		(
			best_box * predictions[... , 26:30] +
			(1 - best_box) * predictions[... ,21:25]
		))
		
		box_predictions[... , 2:4] = torch.sign(box_predictions[... , 2:4]) * torch.sqrt(torch.abs(box_predictions[... , 2:4] + 1e-6)) # 
		box_targets[... , 2:4]     = torch.sqrt(box_targets[... , 2:4])
		
		box_loss = self.mse(
			torch.flatten(box_predictions , end_dim = -2) ,
			torch.flatten(box_targets , end_dim = -2)) 
		
		# object loss (check if its exist)
		pred_box  = (best_box * predictions[... , 25:26] + (1 - best_box) * predictions[... , 20:21])
		obj_loss  = self.mse(
			torch.flatten(exists_box * pred_box) ,
			torch.flatten(exists_box * target[... , 20:21]))

		# class loss
		class_loss = self.mse(
			torch.flatten(exists_box * predictions[... , :20] , end_dim = -2) , 
			torch.flatten(exists_box * target[... , :20] , end_dim = -2)) 

		#no object loss (N,S*S*1)
		noobj_loss = self.mse(
			torch.flatten((1 - exists_box) * predictions[... , 20:21] , start_dim = 1) ,
			torch.flatten((1 - exists_box) * target[... , 20:21] , start_dim = 1)
			)

		noobj_loss += self.mse(
			torch.flatten((1 - exists_box) * predictions[... , 25:26] , start_dim = 1) ,
			torch.flatten((1 - exists_box) * target[... , 20:21] , start_dim = 1)
			)
		# some all losses
		loss = (
				 obj_loss + class_loss
				 self.lambda_coord * box_loss +
				 self.lambda_noobj * nobj_loss +
			   )
		return loss
