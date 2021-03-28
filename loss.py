import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
  def __init__(self , S = 7 , B = 2 , C = 20):
    super(YoloLoss , self).__init__()
    # S * S number of the celles 7 * 7
    # B number of Bounding box 1 
    # C number of classes 20
    self.S   = S
    self.B   = B
    self.C   = C
    # mean square error function
    self.mse = nn.MSELoss(reduction = "sum")
    # obj , nobj
    self.lambda_noobj  = 0.5
    self.lambda_coord  = 5

  def forward(self , predictions , target):
    predictions = predictions.reshape(-1 , self.S , self.S , self.C + self.B * 5)  # N , S , S , 30
    iou_b1 = intersection_over_union(predictions[... , self.C + 1:self.C + 5] , target[... , self.C + 1:self.C + 5]) # num_examples , iou 
    iou_b2 = intersection_over_union(predictions[... , self.C + 6 :self.C + 10] , target[... , self.C + 1:self.C + 5]) # num_examples , iou
    ious   = torch.cat([iou_b1.unsqueeze(0) , iou_b2.unsqueeze(0)] , dim = 0) #
    iou_maxes , best_box = torch.max(ious , dim = 0)
    exists_box  = target[... , self.C].unsqueeze(3)
    box_targets = exists_box * target[..., self.C + 1:self.C + 5]
    box_predictions = exists_box * ( 
		(
			best_box * predictions[... , self.C + 6 :self.C + 10 ] +
			(1 - best_box) * predictions[... ,self.C + 1:self.C + 5]
		))
    box_predictions[... , 2:4] = torch.sign(box_predictions[... , 2:4]) * torch.sqrt(torch.abs(box_predictions[... , 2:4] + 1e-6)) # 
    box_targets[..., 2:4]      = torch.sqrt(box_targets[... , 2:4])
    box_loss = self.mse(
			torch.flatten(box_predictions , end_dim = -2) ,
			torch.flatten(box_targets , end_dim = -2)) 
      # object loss (check if its exist)
    pred_box  = (best_box * predictions[... , self.C + 5 :self.C + 6] + (1 - best_box) * predictions[... , self.C :self.C +1])
    obj_loss  = self.mse(
			torch.flatten(exists_box * pred_box) ,
			torch.flatten(exists_box * target[... , self.C : self.C + 1]))

		# class loss
    class_loss = self.mse(
			torch.flatten(exists_box * predictions[... , :self.C] , end_dim = -2) , 
			torch.flatten(exists_box * target[... , :self.C] , end_dim = -2)) 

		#no object loss (N,S*S*1)
    noobj_loss = self.mse(
			torch.flatten((1 - exists_box) * predictions[... , self.C:self.C +1] , start_dim = 1) ,
			torch.flatten((1 - exists_box) * target[... , self.C:self.C+1] , start_dim = 1)
			)
    
    noobj_loss += self.mse(
			torch.flatten((1 - exists_box) * predictions[... , self.C + 5:self.C + 6] , start_dim = 1) ,
			torch.flatten((1 - exists_box) * target[... , self.C:self.C+1] , start_dim = 1)
			)
		# some all losses
    loss = (
				 obj_loss + class_loss + 
				 self.lambda_coord * box_loss +
				 self.lambda_noobj * noobj_loss )
    return loss
