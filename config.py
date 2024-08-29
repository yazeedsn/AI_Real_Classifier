import sys, os 
sys.path.append(os.path.abspath('C:/Users/Yazeed/Desktop/workspace/flexaibuild'))
from functools import partial

import torch
from torch.nn import Sequential, Dropout, Linear
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.transforms import Resize, ToTensor, Compose


# DEFAULT MODEL
MODEL = mobilenet_v3_small(weights= MobileNet_V3_Small_Weights.DEFAULT)
MODEL.classifier[-1] = Sequential(
    Dropout(),
    Linear(MODEL.classifier[-1].in_features, 2)
)


# trained weights path
WEIGTHS_PATH = "weights/best_weights.pt"

# Inference Transform
INPUT_TRANSFORM = Compose([
  ToTensor(),
  Resize((32, 32)),
  partial(torch.unsqueeze, dim=0),
  ],
)

# label mapping
IDX_TO_CLASS = {0: 'FAKE', 1: 'REAL'}