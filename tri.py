import torch
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
batch_size = 4
from torchvision import transforms
transformsapp = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])

data_dir = 'ISICdataNewAndLovely/jpeg/train/'
foldernames = ['train','rest_of_trainset','val']

def batch_mean_and_sd(loader,fst_moment,snd_moment, cnt):
    runcounter = 0
    print(loader,fst_moment,snd_moment, cnt)
    for images, _ in loader:
        if runcounter %100: print(runcounter)
        print(runcounter)
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
        runcounter +=1 
    return (fst_moment, snd_moment, cnt)

cnt = 0
fst_moment = torch.empty(3)
snd_moment = torch.empty(3)

for foldername in foldernames:
    print(os.path.join(data_dir, foldername))
    dataset = ImageFolder(root = os.path.join(data_dir, foldername), transform=transformsapp)

    loader = DataLoader(
    dataset, 
    batch_size = batch_size, 
    num_workers=4, pin_memory=True)
    (fst_moment, snd_moment, cnt) = batch_mean_and_sd(loader, fst_moment, snd_moment, cnt)

mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        

print("mean and std: \n", mean, std)