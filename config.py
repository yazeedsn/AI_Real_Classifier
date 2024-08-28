import sys, os 
sys.path.append(os.path.abspath('C:/Users/Yazeed/Desktop/workspace/flexaibuild'))
from functools import partial

import torch
from torch.nn import Sequential, Dropout, BatchNorm2d
from torchvision.models import shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights
from torchvision.transforms import Resize, ToTensor, Compose
from flexai.modules import DenseLayer # type: ignore


# DEFAULT MODEL
MODEL = Sequential(
  BatchNorm2d(3),
  shufflenet_v2_x2_0(weights=ShuffleNet_V2_X2_0_Weights.DEFAULT),
)

SHUFFLENET = MODEL[1]

SHUFFLENET.fc = Sequential(
  Dropout(0.7),
  DenseLayer(SHUFFLENET.fc.in_features, 2, act=None),
)



# trained weights path
WEIGTHS_PATH = "weights/best_weights.pt"

# Inference Transform
INPUT_TRANSFORM = Compose([
  ToTensor(),
  Resize((224, 224)),
  #partial(torch.permute,dims=((2,0,1))),
  partial(torch.unsqueeze, dim=0),
  ],
)

# label mapping
IDX_TO_CLASS = {0: 'AI', 1: 'Real'}