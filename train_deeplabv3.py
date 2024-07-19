import torch
import argparse
from torchvision.datasets import VOCSegmentation
from scr.voc_dataset import VOCDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, RandomAffine, ColorJitter, Resize
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import time
def get_args():
    parser = argparse.ArgumentParser(description="Train faster rcnn model")
    parser.add_argument("--data_path", "-d", type=str,default="data", help="Path to dataset")
    parser.add_argument("--year", "-y", type=str, default="2012")
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--num_epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", "-n", type=int, default=6, help="Number of wokers")
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--log_folder", "-p", type=str, default="ObjectDetection/tensorboard",
                        help="Path to generated tensorboard")
    parser.add_argument("--checkpoint_folder", "-c", type=str, default="ObjectDetection/trained_models",
                        help="Path to generated tensorboard")
    parser.add_argument("--saved_checkpoint", "-o", type=str, default=None, help="Continue from this checkpoint")
    # "ObjectDetection/trained_models/last.pt"
    args = parser.parse_args()
    return args

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    model.to(device)
    train_transform = Compose([
        ToTensor(),
        Resize((args.image_size,args.image_size))
    ])
    train_dataset = VOCDataset(root="../data/my_pascal_voc", year="2012", image_set="train", download=False, transform=train_transform, target_transform = train_transform)
    test_dataset = VOCDataset(root="../data/my_pascal_voc", year="2012", image_set="val", download=False, transform=train_transform, target_transform = train_transform)
    train_dataloder = DataLoader(dataset=train_dataset,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=False)
    test_dataloder = DataLoader(dataset=train_dataset,
                                 num_workers=args.num_workers,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    epochs = args.num_epochs
    for epoch in range(epochs):
        for images, labels in train_dataloder:
            images = images.to(device)
            output = model(images)
            masks = output["out"]
            #check values in VOCdataset target - unique
            print(masks.shape)
            time.sleep(3)
if __name__ == '__main__':
    args = get_args()
    main(args)

