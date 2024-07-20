import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scr.voc_dataset import VOCDataset
from torchvision.transforms import ToTensor, Compose, Normalize, RandomAffine, ColorJitter, Resize
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
import argparse
from tqdm import tqdm
import shutil
import time
import os

import warnings

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Train faster rcnn model")
    parser.add_argument("--data_path", "-d", type=str, default="data", help="Path to dataset")
    parser.add_argument("--year", "-y", type=str, default="2012")
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--num_epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", "-n", type=int, default=6, help="Number of wokers")
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--log_path", "-p", type=str, default="tensorboard",
                        help="Path to generated tensorboard")
    parser.add_argument("--checkpoint_path", "-c", type=str, default="checkpoints",
                        help="Path to generated tensorboard")
    parser.add_argument("--saved_checkpoint", "-o", type=str, default=None, help="Continue from this checkpoint")
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    # for name, param in model.named_parameters():
    #     if "aux_classifier" not in name:
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True
    model.to(device)
    transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),  # doi mau buc anh
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = Compose([
        Resize((args.image_size, args.image_size))
    ])
    train_dataset = VOCDataset(root="../data/my_pascal_voc", year="2012", image_set="train", download=False,
                               transform=transform, target_transform=target_transform)
    test_dataset = VOCDataset(root="../data/my_pascal_voc", year="2012", image_set="val", download=False,
                              transform=transform, target_transform=target_transform)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  num_workers=args.num_workers,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=False)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 num_workers=args.num_workers,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 drop_last=False)

    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    writer = SummaryWriter(args.log_path)
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad],  lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    epochs = args.num_epochs
    for epoch in range(epochs):
        # Phase train
        #"Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss:{:.2f} (Coord:{:.2f} Conf:{:.2f} Cls:{:.2f})"
        progress_bar = tqdm(train_dataloader, colour="cyan")
        for iter, (images, targets) in enumerate(progress_bar):
            # Forward
            images = images.to(device)
            targets = targets.to(device)
            targets = torch.squeeze(targets).long()

            output = model(images)
            masks = output['out']
            loss = criterion(masks, targets)
            progress_bar.set_description("Epoch: {}/{} | Iteration: {}/{} | Lr: {} |Loss: {:0.4f}".format(
                                        epoch + 1,
                                        epochs,
                                        iter+1,
                                        len(train_dataloader),
                                        optimizer.param_groups[0]['lr'],
                                        loss))
            writer.add_scalar("Train/Loss", loss, epoch * len(train_dataloader) + iter)
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Test
        all_losses = []
        all_accs = []
        all_ious = []
        model.eval()
        acc_metric = MulticlassAccuracy(num_classes=len(train_dataset.classes)).to(device)
        iou_metric = MulticlassJaccardIndex(num_classes=len(train_dataset.classes)).to(device)
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader, colour="yellow")
            for iter, (images, targets) in enumerate(progress_bar):
                images = images.to(device)
                targets = targets.to(device)
                targets = torch.squeeze(targets, dim=1).long()
                output = model(images)
                masks = output["out"]
                loss = criterion(masks, targets)
                acc = acc_metric(masks, targets)
                iou = iou_metric(masks, targets)

                progress_bar.set_description("Epoch: {}/{} | Iteration: {}/{}| Loss: {:0.2f} | Accuracy: {:0.3f} | IoU: {:0.3f}".format(
                                            epoch + 1,
                                            epochs,
                                            iter + 1,
                                            len(train_dataloader),
                                            loss,
                                            acc.item(),
                                            iou.item()))
                all_losses.append(loss.item())
                all_accs.append(acc.cpu().item())
                all_ious.append(iou.cpu().item())
        loss = np.mean(all_losses)
        acc = np.mean(all_accs)
        iou = np.mean(all_ious)
        writer.add_scalar("Test/Loss", loss, epoch)
        writer.add_scalar("Test/Accuracy", acc, epoch)
        writer.add_scalar("Test/mIoU", iou, epoch)

        checkpoint = {
            "model_state_dict": model.state_dict(),  # all parameter of model, not include architecture
            "epoch": epoch,  # we can know we have trained numbers of epoch
            "optimizer_state_dict": optimizer.state_dict()  # we can see path which we went to
        }


if __name__ == '__main__':
    args = get_args()
    main(args)
