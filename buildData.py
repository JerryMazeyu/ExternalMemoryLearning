from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import opt
from utils import filterDataset

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = CIFAR10(root='./data', download=True, train=True, transform=transform)
test_data = CIFAR10(root='./data', download=True, train=False, transform=transform)

train_data_without_0 = filterDataset(train_data, exclude_classes=[0])

train_dataloader = DataLoader(train_data,batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.num_workers)
test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.num_workers)
