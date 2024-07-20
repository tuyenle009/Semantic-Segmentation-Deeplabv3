import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torch.utils.data import DataLoader
from scr.voc_dataset import VOCDataset
import argparse
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Training deeplabv3 model")
    parser.add_argument("--data_path", "-d", type=str, default="../data/my_pascal_voc", help="this path direct to data")
    parser.add_argument("--batch_size", "-b", type=int, default=6)
    parser.add_argument("--num_workers", "-n", type=int, default=4)
    parser.add_argument("--num_epochs", "-e", type=int, default=100)
    parser.add_argument("--checkpoints_path", "-c", type=str, default="checkpoints")
    parser.add_argument("--log_path", "-l", type=str, default="tensorboard")
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights)
    model.to(device)
    # transform data
    train_transform = Compose([
        ToTensor(),
        Resize(size=(224, 224)),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = Compose([
        ToTensor(),
        Resize(size=(224, 224)),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # load data
    train_dataset = VOCDataset(root=args.data_path, year="2012", image_set="train", transform=train_transform,
                               target_transform=train_transform)
    test_dataset = VOCDataset(root=args.data_path, year="2012", image_set="val", transform=test_transform,
                              target_transform=test_transform)
    # dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False
    )
    test_dataloader = DataLoader(
        dataset=train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )
    #normalize
    epochs = args.num_epochs
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = torch.squeeze(labels).type(torch.LongTensor)
            labels = labels.to(device)
            output = model(images)['out']
            print(output.shape)
            loss = criterion(output,labels).item()
            print(loss)
            # time.sleep(3)




if __name__ == '__main__':
    args = parse_args()
    train(args)
