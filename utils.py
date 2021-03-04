import torch
from collections import Counter

def intersection_over_union(boxes_preds , boxes_labels , format = "corners"):

    if format == "midpoints":
        box1_x1 = boxes_preds[...  , 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[...  , 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[...  , 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[...  , 1:2] + boxes_preds[..., 3:4] / 2
        # Box 2
        box2_x1 = boxes_labels[... , 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[... , 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[... , 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[... , 1:2] + boxes_labels[..., 3:4] / 2

    if format == "corners":  
        box1_x1 = boxes_preds[... , 0:1]
        box1_y1 = boxes_preds[... , 1:2]
        box1_x2 = boxes_preds[... , 2:3]
        box1_y2 = boxes_preds[... , 3:4]
        # box two 2 (x1 , x2 , y1 , y2)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    
    # calculate the width of the inter box
    xes_max = torch.max(box1_x1 , box2_x1)
    xes_min = torch.min(box1_x2 , box2_x2)
    # calculate the height of the inter box
    yes_max = torch.max(box1_y1 , box2_y1)
    yes_min = torch.min(box1_y2 , box2_y2)
    # calculate the area of the intersection box
    intersection = (xes_max - xes_min).clamp(0) * (yes_max - y_min).clamp(0)
    # The area of every box
    box1_area = abs(box1_x2 - box1_x2) * abs(box1_y1 - box1_y2)
    box2_area = abs(box2_x1 - box2_x2) * abs(box2_y1 - box2_y2)
    # calculate the area of the union box
    union = box1_area + box2_area - intersection
    return intersection / (union + 1e-6)

def NMS(bboxes , iou_threshold , prob_threshold , box_format = "corners"):
    assert type(bboxes) == list # [[1 , 0.9 , x1 , y1 , x2 , y2]]
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes , key = lambda x : x[1] , reverse = True)
    boxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes 
        if box[0] != chosen_box[0] or
        intersection_over_union(torch.tensor(chosen_box[2:]) ,
        torch.tensor(box[2:]) , box_format = box_format ) < iou_threshold] 
        boxes_after_nms.append(chosen_box)

    return boxes_after_nms


def mean_average_precision(pred_boxes , true_boxes , iou_threshold = 0.5 , box_format = "midpoints" , num_classes = 20):

    average_precisions  = []
    epsilon = 1e-6
    # For class in classes we calculate the AP
    for c in range(classes):
        # resize work space now for class c
        detections    = [box for box in pred_boxes if box[1] == c] # detection [img_id , ]  some of boxes
        ground_truths = [box for box in true_boxes if box[1] == c] # ground_truths [img_id] some of boxes
        # get dict for all images (index_img : how_many_bbox) 
        amount_boxes  = Counter([gt[0] for gt in ground_truths])
        # convert num of bboxes to tensor of zeros
        for key , value in amount_boxes.items():
            amount_boxes[key] = torch.zeros(val) # dict {img_id , torch.tensor([0 , 0 , 0])}
            
        # sort detection (key of sort is confidence)
        detections.sort(key = lambda x: x[2] , reverse = True)
        # Intialize TRUE POSITIVE AND FALSE POSITIVE
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        # Skip if there is no bboxes to this class
        if total_true_bboxes == 0:
            continue
        for detection_idx , detection in enumerate(detections):
            # collect all ground truth boxes which belong to the same img of the detection
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]] 
            best_iou = 0
            for idx , gt in enumerate(ground_truth_img): # 
                iou = intersection_over_union(torch.tensor(detection[3:]) , torch.tensor(gt[3:]) , box_format = box_format)
                if iou > best_iou:
                    best_iou    = iou
                    best_gt_img = idx

            if best_iou > iou_threshold:
                if amount_boxes[detection[0]][best_gt_img] == 0:
                    TP[detection_idx] = 1
                    amount_boxes[detection[0][best_gt_img]] = 1:
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TF_cumsum  = torch.cumsum(TP , dim = 0)
        FP_cumsum  = torch.cumsum(FP , dim = 0)
        #calculate recall and precison
        recalls    = TF_cumsum / (total_true_bboxes + epsilon)
        precisions = TF_cumsum / (TF_cumsum + FP_cumsum + epsilon)
        recalls    = torch.cat((torch.tensor([0]) ,  recall))
        precisions = troch.cat((torch.tensor([1]) , precisions))
        average_precisions.append(torch.trapz(precisions , recalls))

    return sum(average_precisions) / len(average_precisions)
