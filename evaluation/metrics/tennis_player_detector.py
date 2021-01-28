import torch
import torchvision
import torch.nn as nn
import numpy as np

from utils.tensor_folder import TensorFolder


class TennisPlayerDetector(nn.Module):

    def __init__(self):
        super(TennisPlayerDetector, self).__init__()

        # Loads the model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).cuda()
        self.model.eval()
        self.threshold = 0.8

        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def check_box_boundaries(self, box):
        # Exclude detections in the upper left box
        if box[2] <= 60 and box[1] <= 26:
            return False

        # Exclude detections in the upper right box
        if box[0] >= 200 and box[1] <= 26:
            return False

        # Exclude spectator heads in the bottom
        if box[1] > 80:
            return False

        return True

    def compute_center(self, box):
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def forward(self, observations):
        '''
        Computes the mean squared error between the reference and the generated observations

        :param observations: (bs, observations_count, channels, height, width) tensor with observations
        :return: (bs, observations_count, 2) tensor with x and y coordinates of the detection, -1 if any
        '''

        batch_size = observations.size(0)
        observations_count = observations.size(1)

        # Computes positions one sequence at a time
        all_predicted_centers = []
        for observations_idx in range(observations_count):
            current_observations = observations[:, observations_idx]

            with torch.no_grad():
                predictions = self.model(current_observations)

            for current_prediction in predictions:
                pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(current_prediction['labels'].cpu().numpy())]
                pred_boxes = [(i[0], i[1], i[2], i[3]) for i in list(current_prediction['boxes'].detach().cpu().numpy())]
                pred_score = list(current_prediction['scores'].detach().cpu().numpy())
                filtered_preds = [pred_score.index(x) for x in pred_score if x > self.threshold]
                if len(filtered_preds) > 0:
                    pred_t = filtered_preds[-1]
                    pred_boxes = pred_boxes[:pred_t + 1]
                    pred_class = pred_class[:pred_t + 1]
                else:
                    pred_boxes = []
                    pred_class = []

                #match_found = False
                matches = []
                for idx in range(len(pred_boxes)):
                    if pred_class[idx] == 'person':
                        if self.check_box_boundaries(pred_boxes[idx]):
                            #if match_found:
                                #print("Warning found more than one tennis player, returining the first only")
                            #else:
                                matches.append((pred_boxes[idx][3] - pred_boxes[idx][1], pred_boxes[idx]))
                                #all_predicted_centers.append(self.compute_center(pred_boxes[idx]))
                                #match_found = True
                if len(matches) == 0:
                    all_predicted_centers.append([-1, -1])
                else:
                    if len(matches) > 1:
                        print("Warning found more than one person, returning the tallest detection")
                    # Sort based on the height of the box
                    matches.sort(key=lambda x: x[0])
                    # Uses the highest box between the detected ones
                    all_predicted_centers.append(self.compute_center(matches[-1][-1]))


        predicted_centers = np.asarray(all_predicted_centers).reshape((observations_count, batch_size, 2))
        return np.moveaxis(predicted_centers, 0, 1)



