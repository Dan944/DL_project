"""
This part is used to check the network structure
"""

import torch
from torchsummary import summary

from nets.yolo import YoloBody

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = YoloBody([[3, 4, 5], [1, 2, 3]], 80).to(device)
    summary(m, input_size=(3, 416, 416))
