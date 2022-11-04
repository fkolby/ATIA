#!/usr/bin/env conda
#os.symlink('/content/mnt/My Drive/Colab Notebooks', nb_path)
from torchscan import summary
from argparse import ArgumentParser

import os,sys
from typing import Any, Callable, List, Optional, Type, Union
import torch
from functools import partial
import torch.nn as  nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
import time
import torch
from skimage import io, transform
import PIL
import copy
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import tensorboard
import matplotlib.pyplot as plt
from torch import Tensor
from timm.models.layers import trunc_normal_, DropPath
#from timm.models.registry import register_model
import torchvision.models._api
import torchvision
#these gave troubles:
#import torchvision.transforms._presets
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import Weights, WeightsEnum #,register_model
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.datasets import ImageFolder
#####

from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing
from timm.data.auto_augment import rand_augment_transform
from timm.loss import SoftTargetCrossEntropy

#from timm.data.dataset import ImageDataset
#from timm.data.parsers.parser_image_folder import ParserImageFolder
from torchvision import datasets, transforms, models
import torchmetrics
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics.classification import BinaryF1Score,BinaryAUROC, BinaryConfusionMatrix
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import DeviceStatsMonitor






def add_arguments():
    parser = ArgumentParser()
    parser.add_argument("--seed", dest="seed", type = int, default = 442)
    parser.add_argument("--data_dir", dest = "data_dir", type=str, default='')#default on comp: ISICdataNewAndLovely/jpeg/train/
    parser.add_argument("--max_epochs", dest = "maxEpochs", type = int, default = 2)
    parser.add_argument("--batchSize", dest = "batchSize", type = int, default = 2)
    parser.add_argument("--modeltype", dest = "modeltype", type = str, default = "baseCase")
    parser.add_argument("--maxTime", dest = "maxTime", type = int, default = -1)
    parser.add_argument("--logDiraddendum", dest = "logDir", type = str, default = "")
    parser.add_argument("--gpu", dest = "gpu", type = bool, default = False)
    parser.add_argument("--workers", dest = "workers", type = int, default = 4)
    parser.add_argument("--devices", dest = "devices", type = int, default = 0)
    parser.add_argument("--testSetName", dest = "testSetName", type = str, default = "val")
    parser.add_argument("--portName", dest = "port", type = int, default = 36159)
    return parser



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        #self.apply(self._init_weights)
    def _init_weights(self,m):
      if isinstance(m, (nn.Conv2d, nn.Linear)):
          trunc_normal_(m.weight, std=.02)
          if m.bias is not None:
            print(type(m.bias))
            nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        #self.apply(self._init_weights)
 
    def _init_weights(self,m):
      if isinstance(m, (nn.Conv2d, nn.Linear)):
          trunc_normal_(m.weight, std=.02)
          if m.bias is not None:
            print(type(m.bias))
            nn.init.constant_(m.bias, 0)
            

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(pl.LightningModule):
    def __init__(
        self,
        args,
        ds,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        epochs: int,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        l1kernelSize = 7,
        l1stride = 2,
        l1pad = 3
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.ds = ds
        self.args = args
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=l1kernelSize, stride=l1stride, padding=l1pad, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if self.args.modeltype != "Patch": self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        #heron
        self.save_hyperparameters()
        self.lr = 0.0004#0.03630#0.004#global_batch_size/4096#selflearn
        self.epochs = epochs
        self.newImgs = []
        #self.apply(self._init_weights)
        self.watch = time.time()
        timer_ = 0
        self.log("timer_", timer_,sync_dist=True)
    def _init_weights(self,m):
      if isinstance(m, (nn.Conv2d, nn.Linear)):
          trunc_normal_(m.weight, std=.02)
          if m.bias is not None:
            print(type(m.bias))
            nn.init.constant_(m.bias, 0)
        #notthis
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        print(self.groups)
        print("#######################################/n","importantes: /n/n/n/n/n/n",self.inplanes, downsample, self.groups, self.base_width,block.expansion, block, planes, blocks, stride, dilate, previous_dilation,"/n#######################################/n")
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        print(layers)
        self.inplanes = planes * block.expansion
        print(self.inplanes)
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)
      
    def _getdvc(self):
      return torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.args.modeltype != "Patch": x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    #heronagain

    def mixup_fn(self,x,y):
      mix = Mixup(mixup_alpha = 0.8, cutmix_alpha = 1.0, cutmix_minmax = None, prob = 1, switch_prob = 0.5,mode = "batch",label_smoothing = 0.1,num_classes=2)
      if x.size(dim=0)%2 != 0:
        outputx,outputy = mix(x[1:],y[1:])
        xnaught = torch.reshape(x[0], shape = (1,3,224,224))
        yngt = torch.reshape(y[0].clone().detach(),shape=(1,1))
        ynaught = torch.cat((yngt,yngt),dim=1)
        (outputx,outputy) = (torch.cat((xnaught,outputx),dim=0).to(self._getdvc()),torch.cat((ynaught,outputy),dim=0).to(self._getdvc()))
      else: 
        outputx,outputy = mix(x,y)  
      return (outputx,outputy)
    def training_step(self, batch, batch_idx):
      criterion=nn.CrossEntropyLoss()
      x,y = batch
      if self.args.gpu == True:
        x,y = self.mixup_fn(x,y) 

      #x = RandomErasing(probability=0.25, mode="pixel",max_count=1)(x)
      self.stepmax = len(x)*self.epochs
      y_hat = self(x)
      #print("train, y:", y[0:3], "yhat", y_hat[0:3])
      loss = criterion(y_hat,y)
      #device = self._getdvc()
      #acc_fn = torchmetrics.Accuracy(threshold=0.5).to(device)
      #accuracy = acc_fn(torch.argmax(y_hat,dim=1),y).to(device)
      self.log("train_loss", loss, on_step= True,on_epoch = True, prog_bar = True, logger=True,sync_dist=True)
      return loss
    def validation_step(self, batch, batch_idx):
      criterion=nn.CrossEntropyLoss()
      x,y = batch
      y_hat = self(x)
      y_hatacc = torch.softmax(y_hat,dim=1)[:,1]
      #print("val, y:", y[0:3], "yhat", y_hat[0:3], "yhatacc", y_hatacc[0:3])
      loss = criterion(y_hat,y) 
      device = self._getdvc()
      acc_fn = torchmetrics.Accuracy(threshold=0.5).to(device)
      accuracy = acc_fn(y_hatacc,y)
      #auroc = BinaryAUROC().to(device)
      #confusion = BinaryConfusionMatrix().to(device)
      #F1 =  BinaryF1Score().to(device)
      #aurocstat = auroc(y_hatacc, y)
      #print(confusion(y_hatacc,y))
      #self.log("F1_score", F1(y_hatacc,y))
      timer_ = time.time()-self.watch
      if timer_ > self.args.maxTime:
        if timer_ < 0:
          timer_ += -1
        else: timer_ = -1
      self.log("timer_", timer_,sync_dist=True)
      print(timer_)
      self.log("val_loss", loss,sync_dist=True)
      self.log("accuracy",accuracy,sync_dist=True)
      #self.log("AUROC", aurocstat)
      self.newImgs.append(im_convert(x))
      grid = torchvision.utils.make_grid(im_convert(x))
      self.logger.experiment.add_image('generated_images', grid, 0) 
      return loss

    
    #workers = args.workers # dummy
    def test_step(self, batch, batch_idx):
      criterion=nn.CrossEntropyLoss()
      x,y = batch
      y_hat = self(x)
      loss = criterion(y_hat,y)
      grid = torchvision.utils.make_grid(x) 
      self.logger.experiment.add_image('generated_images', grid, 0) 
      self.log("test_loss", loss,sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
      x,y = batch
      y_hat = self(x)
      yhatacc = torch.softmax(y_hat,dim=1)[:,1]
      return yhatacc
    def configure_optimizers(self):
      num_training_steps_per_epoch = len(self.ds['train'])/self.args.maxEpochs
      opt = optim.AdamW(self.parameters(), lr=self.lr,betas = (0.9,0.999))
      T_max= num_training_steps_per_epoch*self.epochs
      learnRate = self.lr*(1/4)*0.001
      #base_value=4e-3, final_value=1e-6, epochs, niter_per_ep, warmup_epochs=4,
      #               start_warmup_value=0, warmup_steps=-1):
      sched = ExponentialLR(opt,0.9)
      #sched = CosineAnnealingLR(opt, T_max=T_max,eta_min=learnRate)
      return {"optimizer": opt, "lr_scheduler": sched}
    
    def train_dataloader(self):
        
        return torch.utils.data.DataLoader(self.ds['train'], batch_size = self.args.batchSize,
                                             shuffle=True, num_workers=args.workers, pin_memory=True)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.ds['val'], batch_size = self.args.batchSize,
                                              shuffle=False, num_workers=self.args.workers, pin_memory=True)
    
    #nomore

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
  


#picture_dir = data_dir+'/balanced_trainset'
#if not os.path.exists('/content/lightning_logs'): os.makedirs('/content/lightning_logs')
#channels = 3?
def ResNet50(num_classes, args,ds,epochs, layers, batch_size, l1kernelSize = 7,l1stride = 2, l1pad = 3):
    return ResNet(args, ds, Bottleneck, layers,epochs, num_classes, batch_size, l1kernelSize = l1kernelSize, l1stride=l1stride, l1pad=l1pad)
#resnet50_model = ResNet50(num_classes=2, block=[3,4,6,3],epochs = 2, batch_size = global_batch_size, l1pad = 3, l1stride = 2, l1kernelSize = 7)
#print(resnet50_model)



#from torchvision.
def data_transforms():
    mean = [0.8306, 0.6337, 0.5970]
    std = [0.1541, 0.1970, 0.2245]
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform(config_str = "rand-m9-mstd0.5-inc1", hparams = dict(translate_const=250, img_mean=(int(255*mean[0]),int(255*mean[1]),int(255*mean[2])))),#,hparams=dict(translate_const=250,img_mean=(128, 128, 128)))
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    }
    return data_transforms


#print(args.seed)
#print(data_dir)
#global_batch_size = 0 #dummy

# Helper function to unnormalize and plot images
def im_convert(lsttensor):
    """ Display a tensor as an image. """
    mean = [0.8306, 0.6337, 0.5970]
    std = [0.1541, 0.1970, 0.2245]
    imgs = lsttensor[:8,:,:,:].to("cpu")
    imgs = imgs*torch.tensor(np.array((std[0], std[1],std[2])).reshape(3, 1, 1)) + torch.tensor(np.array((mean[0], mean[1], mean[2])).reshape(3, 1, 1))
    return imgs


#randaugment
#mixup
#cutmix
#random erasing
#label smoothing

#print(os.path.join(data_dir, 'test'))

#dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#class_names = image_datasets['train'].classes


#openAiCALC:


#add-multiplies per forward pass) * (2 FLOPs/add-multiply) * (3 for forward and backward pass) * (number of examples in dataset) * (number of epochs)


#print("set workers to: ", workers)



# use this to calculate epoch: https://openai.com/blog/ai-and-compute/
#maxepochs=2
#max_epochs = args.maxEpochs

print("got here")

#lav folder med logs og dato i drive.


#global_batch_size = args.batchSize#64#256
#max_time = args.maxTime
#workers = args.workers# int(os.cpu_count())

def imgData():
        image_datasets = {x: ImageFolder(root = os.path.join(args.data_dir, x),
                                        transform = data_transforms()[x])
                for x in ['train', 'val','test']}
        return image_datasets

#do testing before predicting so as to include loss metrics?





def main(args):
    earlystopEpochs =40
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    dvcStats = DeviceStatsMonitor(cpu_stats=True)
    logpath = args.modeltype + "-" + args.logDir + "/"
    EarlyStopCB_Time = EarlyStopping('timer_', patience = 0, mode="max")
    
    curtim = time.strftime("%c")
    os.makedirs(logpath+"lightning_logs/"+curtim, exist_ok=True)
    logdir = logpath+"lightning_logs/"
    #tb = tensorboard.program.TensorBoard()
    #tb.configure(argv = [None, f'--logdir={logdir}',f"--port={args.port+args.seed}"])
    #url = tb.launch()
    #print(f"Tensorboard listening on {url}")
    

    ds = imgData()

    tb_logger = pl_loggers.TensorBoardLogger(save_dir = logpath+"lightning_logs/", name = args.modeltype, version = str(args.seed) + "-" + curtim)
    if args.gpu == True:
        trainer = pl.Trainer(accelerator = "gpu", profiler  = "simple", devices = args.devices, max_epochs = args.maxEpochs, log_every_n_steps = 1, callbacks = [dvcStats,lr_monitor,checkpoint_callback,EarlyStopCB_Time , TQDMProgressBar(refresh_rate=3)], logger = tb_logger)
    else:
        trainer = pl.Trainer(max_epochs = args.maxEpochs, profiler = "simple", log_every_n_steps = 1,callbacks = [dvcStats,lr_monitor,checkpoint_callback,EarlyStopCB_Time ], logger = tb_logger) #, TQDMProgressBar(refresh_rate=3)]
    if args.modeltype == "baseCase":
        
        timestart = time.time()
        flopCalcedModelBase = ResNet50(num_classes=2, args=args,ds=ds,layers=[3,3,9,3],epochs = args.maxEpochs, batch_size = args.batchSize, l1pad = 3, l1stride = 2, l1kernelSize = 7)
        summary(flopCalcedModelBase, (3,224,224), max_depth=2)
        model = flopCalcedModelBase
        trainer.fit(model)
        modelRunTime = time.time()-timestart   

    elif args.modeltype == "4x4":
        flopCalcedModelLowKern = ResNet50(num_classes=2,args=args,ds=ds, layers=[3,3,9,3],epochs = args.maxEpochs, batch_size = args.batchSize, l1pad = 1, l1stride = 2, l1kernelSize = 4)
        summary(flopCalcedModelLowKern, (3,224,224), max_depth=2)
        timestart= time.time()
        model = flopCalcedModelLowKern
        
        trainer.fit(model)
        modelRunTime = time.time()-timestart   
    elif args.modeltype == "Patch": 
        flopCalcedModelPatch = ResNet50(num_classes=2,args=args,ds=ds, layers=[3,3,9,3],epochs = args.maxEpochs, batch_size = args.batchSize, l1pad = 0, l1stride = 4, l1kernelSize = 4)
        summary(flopCalcedModelPatch, (3,224,224), max_depth=2)
        timestart= time.time()
        model = flopCalcedModelPatch
        
        trainer.fit(model)
        modelRunTime = time.time()-timestart
    else:
        print("model not found")
        exit()
    
    predloader = torch.utils.data.DataLoader(ds[args.testSetName], batch_size = args.batchSize,
                                                shuffle=False, num_workers=args.workers, pin_memory=True)
    preds = trainer.predict(model,predloader)


    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import RocCurveDisplay
    ysl = torch.tensor([])
    for el in iter(predloader):
        (_,ys) = el
        ysl = torch.cat((ysl,ys), dim=0)
    psl = torch.tensor([])
    for p in preds:
        psl = torch.cat((psl,p),dim=0)

    print(ysl)
    print(psl)
    #for rl in torch.vstack((ysl, psl)).t():
    #  print(rl)
    print(roc_auc_score(ysl, psl))

    with open(logpath+'lightning_logs/'+'/params.txt', 'a') as f:
        f.write("\n" + "TIME: " + str(curtim) + "\n")
        f.write("VARS: " + str(vars(args)) + "\n")
        if args.gpu:
            for i in range(torch.cuda.device_count()):
                f.write("Device, number: " + str(i) + ", name: " + torch.cuda.get_device_name(i) +"\n")
        f.write("Time to run model:" +str(modelRunTime) + "\n")
        f.write("AUC: " + str(roc_auc_score(ysl, psl)) + "\n")
        ysl = ysl.int()
        f.write("ACC: " + str(torchmetrics.Accuracy(threshold=0.5).to(torch.device("cpu"))(psl, ysl)) + "\n")
        f.write("################################################################################\n")
    disp = RocCurveDisplay.from_predictions(ysl,psl)
    plt.savefig(logpath+"lightning_logs/" + curtim)
    return 0

if __name__ =="__main__":
    parser = add_arguments()
    args = parser.parse_args()
    if args.gpu==True and args.devices == 0:
        print("specify number of devices")
        exit()
    if args.gpu==True and args.batchSize ==2:
        print("specify batchsize")
        exit()
    pl.seed_everything(seed=args.seed,workers=True) #does not -in fact seed everything(due to timm)
    main(args)
