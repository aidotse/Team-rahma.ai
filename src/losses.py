from fastai.vision.all import *
from fastai.vision import models
import torch
import torchvision

import sys
sys.path.append('./')
from src.models.loss_model import get_loss_model_vgg, siamese_splitter
from src.types.tensor_tiles import BrightfieldTile, FluorescenceTile

class TileMSELoss(torch.nn.Module): 
    """ MSE loss for custom tile classes """
    def __init__(self,
                 reduction: str = 'mean',
                 denormalized:bool=False,
                 denorm_mean:float=None,
                 denorm_std:float=None
                ) -> None:
        super(TileMSELoss, self).__init__()
        self.reduction = reduction
        self.__name__ = 'MSE'
        if denormalized and denorm_std is not None and denorm_mean is not None:
            self.denormalized = denormalized
            self.denorm_mean, self.denorm_std = tensor(np.array(denorm_mean)), tensor(np.array(denorm_std))
        else:
            self.denormalized = False

    def forward(self, input, target) -> Tensor:
        if self.denormalized:
            # to channel last format
            _input = input.permute(0,2,3,1)
            _target = target.permute(0,2,3,1)
            
            # denormalize
            _input = (_input * self.denorm_std.to(_input.device)) + self.denorm_mean.to(_input.device)
            _target = (_target * self.denorm_std.to(_target.device)) + self.denorm_mean.to(_target.device)
            
            # to channel first format
            _input = _input.permute(0,3,1,2)
            _target = _target.permute(0,3,1,2)
            
            return F.mse_loss(TensorImage(_input), TensorImage(_target), reduction=self.reduction)
        else:
            return F.mse_loss(TensorImage(input), TensorImage(target), reduction=self.reduction)

class TileL1Loss(torch.nn.Module):
    """ L1 loss for custom tile classes """
    def __init__(self, 
                 reduction: str = 'mean',
                 denormalized:bool=False,
                 denorm_mean:float=None,
                 denorm_std:float=None
                ) -> None:
        super(TileL1Loss, self).__init__()
        self.reduction = reduction
        self.__name__ = 'MAE'
        if denormalized and denorm_std is not None and denorm_mean is not None:
            self.denormalized = denormalized
            self.denorm_mean, self.denorm_std = tensor(np.array(denorm_mean)), tensor(np.array(denorm_std))
        else:
            self.denormalized = False

    def forward(self, input, target) -> Tensor:
        if self.denormalized:
            # to channel last format
            _input = input.permute(0,2,3,1)
            _target = target.permute(0,2,3,1)
            
            # denormalize
            _input = (_input * self.denorm_std.to(_input.device)) + self.denorm_mean.to(_input.device)
            _target = (_target * self.denorm_std.to(_target.device)) + self.denorm_mean.to(_target.device)
            
            # to channel first format
            _input = _input.permute(0,3,1,2)
            _target = _target.permute(0,3,1,2)
            
            return F.l1_loss(TensorImage(_input), TensorImage(_target), reduction=self.reduction)
        else:
            return F.l1_loss(TensorImage(input), TensorImage(target), reduction=self.reduction)
    
class Chan_MSE(torch.nn.Module):
    """ Single channel MSE metric for tile classes """
    def __init__(self, reduction: str = 'mean', chan:int=0) -> None:
        super(Chan_MSE, self).__init__()
        self.reduction = reduction
        self.chan = chan
        assert chan > -1 and chan < 3, 'chan must be 0,1 or 2'
        self.__name__ = f'{chan}_MSE'

    def forward(self, input, target) -> Tensor:
        return F.mse_loss(TensorImage(input[self.chan]), TensorImage(target[self.chan]), reduction=self.reduction)

class VGGPerceptualLoss(torch.nn.Module):
    """ Perceptual loss with conv blocks from imagenet pre-traing VGG16 """
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
                
        self.blocks = torch.nn.ModuleList(blocks).cuda()
        
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).cuda()
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).cuda()
        self.resize = resize
        
        self.mse = TileMSELoss()

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            #loss += torch.nn.functional.l1_loss(x, y)
            loss += self.mse(x, y)
        return loss
    
class VGGTrainedPerceptualLoss(torch.nn.Module):
    """ Perceptual loss with conv blocks from a VGG16 model loaded from a given path. """
    def __init__(self, pretrained_path, add_l1=True):
        super(VGGTrainedPerceptualLoss, self).__init__()
        
        def loss_model():
            model = get_loss_model_vgg()
            model.load_state_dict(torch.load(pretrained_path))
            return model
        
        model = loss_model()
        self.L1 = TileL1Loss()
        self.add_l1 = add_l1
        
        blocks = []
        blocks.append(model.encoder[0][:4].eval())
        blocks.append(model.encoder[0][4:9].eval())
        blocks.append(model.encoder[0][9:16].eval())
        blocks.append(model.encoder[0][16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        
        # compensate layer output size differences by weights
        # all layer blocks affect equally
        self.block_weights = [1,2,4,4]
        
        # switch relus to inplace = False
        blocks[0][2] = torch.nn.ReLU(inplace=False)
        blocks[1][1] = torch.nn.ReLU(inplace=False)
        blocks[2][0] = torch.nn.ReLU(inplace=False)
        blocks[2][3] = torch.nn.ReLU(inplace=False)
        blocks[3][0] = torch.nn.ReLU(inplace=False)
        blocks[3][3] = torch.nn.ReLU(inplace=False)
        blocks[3][6] = torch.nn.ReLU(inplace=False)
        
        # freeze all batchnorm layers
        self.blocks = torch.nn.ModuleList(blocks).cuda()
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                m.eval()
                m.affine = False
                m.track_running_stats = False
        self.blocks.apply(set_bn_eval)

    def forward(self, input, target):
        x = input
        y = target
        # add L1 with same weight combination = affects equally as perceptual part
        loss = self.L1(x, y) * np.array(self.block_weights).sum() if self.add_l1 else 0.0
        for block, weight in zip(self.blocks, self.block_weights):
            x = block(x)
            y = block(y)
            loss += weight * self.L1(x, y)
        return loss
