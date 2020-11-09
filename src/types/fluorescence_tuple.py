from fastai.vision.all import *
import torch

class FluorescenceTuple(fastuple):

    def show(self, ctx=None, **kwargs):
        img1,img2,similarity = self
        
        def equalize(img):
            if not isinstance(img, Tensor):
                t = tensor(img)
                t = t.permute(2,0,1)
            else: 
                t = img
                
            t = t.permute(1,2,0)
            
            # cover the full 0-1 range for visualization
            t = (t + 2.) / 4.0
            t = t.clamp(0.,1.).permute(2,0,1)
            return t

        t1,t2 = equalize(img1), equalize(img2)
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1,line,t2], dim=2), title=similarity, ctx=ctx, **kwargs)