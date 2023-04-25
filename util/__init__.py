"""Common Constants"""
from torchvision import transforms

# mean and standard deviation of channels of CIFAR-10 images
MEAN = [x / 255 for x in [125.3, 123.0, 113.9]]
STD = [x / 255 for x in [63.0, 62.1, 66.7]]

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)
TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

TINY_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)
