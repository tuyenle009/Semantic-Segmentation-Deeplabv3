from torchvision.datasets import VOCSegmentation
from torchvision.transforms import ToTensor
class VOCDataset(VOCSegmentation):
    def __init__(self, root, year="2012", image_set="train", download="False", transform= None,target_transform=None, size=224):
        super().__init__(root, year, image_set, download, transform, target_transform)
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                           'train', 'tvmonitor']
        self.size = size
    def __getitem__(self, item):
        image, target = super().__getitem__(item)
        # if transform:
        #     image = transform(image)
        #     target = transform(target)
        return image, target
if __name__ == '__main__':
    transform = ToTensor()
    dataset = VOCDataset(root="../../data/my_pascal_voc", year="2012", image_set="train", transform= transform,download=False, size=224)
    img, label = dataset[200]
    print(img)
    print(label)