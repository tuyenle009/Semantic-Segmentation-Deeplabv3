import torch
import argparse
from torchvision.datasets import VOCSegmentation
import cv2

def get_args():
    parser = argparse.ArgumentParser("Train deeplab v3")
    parser.add_argument("--data_path","-d",type=str, default="my_pascal_voc")
    parser.add_argument("--batchsize","-b", type=int, default= 8)
    parser.add_argument("--lr","-l", type=float, default= 0.01)
    parser.add_argument("--epochs","-e", type=int, default=100)
    args = parser.parse_args()
    return args

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    model.to(device)
    train_dataset = VOCSegmentation(root="../data")
if __name__ == '__main__':
    args = get_args()

