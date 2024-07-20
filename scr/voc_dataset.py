import numpy as np
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import ToTensor
from torchvision.transforms import ToTensor, Compose, Normalize, Resize


class VOCDataset(VOCSegmentation):
    def __init__(self, root: str, year: str = "2012", image_set: str = "train", download: bool = False, transform=None,
                 target_transform=None, transforms=None):
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']

    def __getitem__(self, item):
        image, target = super().__getitem__(item)
        target = np.array(target)
        target[target==255] =0
        return image, target
if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize(size=(224, 224)),
    ])
    dataset = VOCDataset(root="../../data/my_pascal_voc", year="2012", image_set="train", transform= transform,target_transform=transform)
    img, label = dataset[200]
    print(img)
    print(label)