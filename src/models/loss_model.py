from fastai.vision.all import *
from fastai.vision import models
import torch

class SiameseModel(Module):
    def __init__(self, encoder, head):
        self.encoder,self.head = encoder,head
    
    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs)

def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]

def get_loss_model_vgg():
    encoder = create_body(vgg16_bn, cut=-2)
    head = create_head(2048,  2, ps=0.5)
    model = SiameseModel(encoder, head)
    return model