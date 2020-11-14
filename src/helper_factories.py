from fastai.vision.all import *
from fastai.vision import models
import torch
from functools import partial
import os
import json
from glob import glob
from tqdm.auto import tqdm
import time

sys.path.append('./')
sys.path.append('../')
from src.losses import *
from src.models.unet_models import *
from src.types.tensor_tiles import BrightfieldTile, FluorescenceTile
from src.types.brightfield_slide import BrightFieldSlide
from src.types.fluorescence_slide import FluorescenceSlide
from src.types.parent_slide import SlideTile, SlideImage
from src.stats_reader import DataStats

def get_learner(
        base_arch:str, 
        dls, 
        pretrained_path=None, 
        loss_method='mse', 
        perceptual_pth=None, 
        target_mean=None,
        target_std=None,
    ):
    """
    Returns a learner loaded with specified model and weights.
    
    Arguments:
        base_arch       (str): identifier for the base architecture. one of 'resnet50', 'resnest50' 
        dls      (Dataloader): Fastai dataloader instance
        pretrained_path (str): if not None (default), weights are loaded from this .pth path
        loss_method     (str): option for loss, one of mse, l1, perceptual, perceptual_trained
        perceptual_pth  (str): if loss_method is perceptual_trained, this should point to vgg loss model's .pth file
        target_mean    (list): denormalization mean for losses
        target_std     (list): denormalization std for losses
    """
    
    base = get_base_arch(base_arch)
    loss = get_loss(loss_method, perceptual_pth, target_mean, target_std)
    
    learn = unet_learner(
        dls, 
        base,
        pretrained=True,
        n_out=3, 
        loss_func=loss, 
        metrics=[TileMSELoss(), Chan_MSE(chan=0), Chan_MSE(chan=1), Chan_MSE(chan=2)]
    )
    
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path)
        learn.model.load_state_dict(state_dict)
    
    return learn

def get_unet(base_arch:str, size=tuple([256,256]), n_out=3, weights_path=None, device='cuda'):
    """
    Initialize Unet model from base arch and input size.
    Optionally, provide path to model weights.
    
    Arguments:
        base_arch    (str): identifier for the base architecture. one of 'resnet50', 'resnet34', 'resnest50, 'efficientnetb5' 
        size   (tuple:int): input size tuple, default=(256,256)
        n_out        (int): default=3, output channels
        weights_path (str): state dict *.pth file path, default=None
        device       (str): 'cpu' or 'cuda'
        
    Returns:
        unet_model (model): unet model loaded with imagenet or specified weights
    """
    
    # construct the unet model
    _default_meta = {'cut':None, 'split':default_split}
    base = get_base_arch(base_arch)
    meta = model_meta.get(base, _default_meta)
    pretrained = weights_path is None
    body = create_body(base, 3, pretrained , meta['cut']) # in_size=3 for pretrained imagenet weights - doesn't affect model
    unet_model = models.unet.DynamicUnet(body, n_out, size)
    
    # load trained weights
    if weights_path is not None:
        state_dict = torch.load(weights_path)
        unet_model.load_state_dict(state_dict)
    
    # device placement
    if device == 'cpu':
        unet_model.cpu()
    else:
        unet_model.cuda()
    
    return unet_model

def get_base_arch(base_arch:str):
    """ 
    Returns the corresponding base arch func. for a given identifier' 
    
    Arguments:
        base_arch         (str): identifier for the base architecture. one of 'resnet50', 'resnest50'
        
    Returns:
        model_func   (callable): model func that returns a base architecture 
    """
    
    if base_arch == 'resnet50':
        base = resnet50_7chan
    elif base_arch == 'resnest50':
        base = resnest50_7chan
    elif base_arch == 'resnet34':
        base = resnet34_7chan
    elif base_arch == 'efficientnetb5':
        base = efficientnetb5_7chan
    else:
        raise Exception('not implemented', f'{base_arch} base_arch not implemented')
    return base

def get_loss(loss_method:str, perceptual_pth=None, target_mean=None, target_std=None,):
    """ 
    Returns the corresponding loss instance for a given loss method identifier 
    
    Arguments:
        loss_method    (str): option for loss, one of mse, l1, perceptual, perceptual_trained
        perceptual_pth (str): weights path for trained perceptual loss. This must be provided if loss_method=perceptual_trained
        target_mean   (list): denormalization mean for losses
        target_std    (list): denormalization std for losses
        
    Returns:
        loss (loss callable): loss instance that can be used as loss_func
    """
    if loss_method == 'mse':
        loss = TileMSELoss(denormalized=(target_mean is not None), denorm_mean=target_mean, denorm_std=target_std)
    elif loss_method == 'perceptual_trained':
        loss = TileL1Loss(denormalized=(target_mean is not None), denorm_mean=target_mean, denorm_std=target_std)
    elif loss_method == 'perceptual':
        loss = VGGPerceptualLoss()
    elif loss_method == 'perceptual_trained':
        loss = VGGTrainedPerceptualLoss(pretrained_path=perceptual_pth)
    else:
        raise Exception('not implemented', f'{loss_method} loss_method not implemented')
    return loss

def get_inference_func(model_dir):
    """
    Returns a callable for inference.
    
    Arguments:
        model_dir           (str): Directory containing subfolders for models and each subdir includes zoom_20, zoom_40, zoom_60 dirs
        
    Returns:
        inference_func (callable): See callable Arguments and Returns below.
        
    -------------------------
    inference_func
    
    Arguments:
        zoom            (int): magnification. one of 20,40,60
        predict_files  (list): List of tiff file paths to predict. Each complete slide has 7 brightfield tiffs. Either predict_files or predict_dir must be provided.
        predict_dir     (str): Directory containing *C04.tif files to predict. Each complete slide has 7 brightfield tiffs. Either predict_files or predict_dir must be provided.
        use_perceptual_loss_model (bool): Default=True. If False, select only-mse-trained model if available
        
    Returns:
        FluorescenceSlides ([[FluorescenceSlide]]): Nested list of FluorescenceSlides. Outer list has n-slides items and inner list has n-models items.
    """
    
    def inference_func(
        model_dir,
        zoom:int,
        predict_files=None, 
        predict_dir=None,
        output_dir=None, 
        use_perceptual_loss_model=True,
        disable_prints=False,
        bs=16
    ):
        
        assert not(predict_files is None and predict_dir is None), 'Please provide either predict_files or predict_dir'
        assert not(predict_files is not None and predict_dir is not None), 'Please provide only one. predict_files or predict_dir'
    
        # load prediction input files
        if predict_dir is not None:
            input_files = glob.glob(predict_dir + '/*C04.tif')
        else:
            input_files = predict_files
        assert len(input_files) % 7 == 0, 'length of input file list is not divisible by 7'
        
        # check that the input files are brightfield files
        for input_fn in input_files:
            assert str(input_fn).endswith('C04.tif'), f'input file {input_fn} does not end with C04.tif'

        # load model paths and configs for the zoom
        model_paths, model_config_paths = [], []
        for sub_dir in [_dir for _dir in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, _dir))]:
            contents = os.listdir(os.path.join(model_dir, sub_dir))
            if f'zoom_{zoom}' in contents:
                dir_files = os.listdir(os.path.join(model_dir, sub_dir, f'zoom_{zoom}'))
                if 'model_mse_trained.pth' in dir_files and not use_perceptual_loss_model:
                    pth_file = os.path.join(model_dir, sub_dir, f'zoom_{zoom}', 'model_mse_trained.pth')
                else:
                    pth_file = os.path.join(model_dir, sub_dir, f'zoom_{zoom}', 'model.pth')
                config_file = os.path.join(model_dir, sub_dir, f'zoom_{zoom}', 'model_config.json')
                if os.path.isfile(pth_file) and os.path.isfile(config_file):
                    model_paths.append(pth_file)
                    model_config_paths.append(config_file)
        
        assert len(model_paths) > 0, 'no valid models found'
        
        unique_slide_names = np.unique(np.array([(os.path.basename(fn).split('.')[0][:-9]) for fn in input_files]))
        fluorescence_slides_list = []
        
        unet_models = []
        for model_path, model_config_path in zip(model_paths, model_config_paths):
            # read config
            with open(model_config_path) as json_file:
                config = json.load(json_file)

            tile_sz = config['tile_sz']

            # construct unet with weights
            unet_model = get_unet(
                base_arch=config['base_arch'],
                size=tuple([tile_sz, tile_sz]),
                n_out=3,
                weights_path=model_path,
                device='cuda'
            )
            unet_model.eval()
            unet_models.append(unet_model)
        
        
        inference_start_time = time.time()
        for unique_slide in tqdm(unique_slide_names, desc='[inference] predicting slides', disable=disable_prints):
            slide_input_files = [fn for fn in input_files if unique_slide in fn]
            assert len(slide_input_files) == 7, f'slide has {len(slide_input_files)} tif files instead of 7.'

            # construct slide
            brightfield_slide = BrightFieldSlide.fromFiles(fns=slide_input_files, name=os.path.basename(unique_slide))
            fluorescence_slides = []

            for unet_model, model_config_path in zip(unet_models, model_config_paths):
                # read config
                with open(model_config_path) as json_file:
                    config = json.load(json_file)

                tile_sz = config['tile_sz']
                tile_overlap = config['tile_overlap']
                STATS = DataStats(stats=config['stats'])

                # split to tiles with tile size and overlap
                brightfield_tiles = brightfield_slide.getTiles(tile_sz, tile_overlap)
                fluorescence_tiles = []
                
                # batched inference
                
                #for tile_index in range(0, len(brightfield_tiles), bs):
                #    tile_batch = brightfield_tiles[tile_index:min(tile_index + bs, len(brightfield_tiles) - 1)]
                #    br_img_batch = torch.stack([BrightfieldTile.from_numpy(tile.img, stats=STATS) for tile in tile_batch])
                #    br_batch = tensor(br_img_batch).reshape(-1, *tensor(br_img_batch).shape[-3:]).cuda()
                #    pred_fs_batch = unet_model(br_batch).detach().cpu()
                #    pred_img_batch = [FluorescenceTile(pred_fs).to_numpy(stats=STATS) for pred_fs in pred_fs_batch]

                #    # create a SlideTile
                #    fs_tile_batch = tile_batch
                #    for fs_tile, pred_img in zip(fs_tile_batch, pred_img_batch):
                #        fs_tile.img = pred_img
                #    fluorescence_tiles += fs_tile_batch
                
                for tile in brightfield_tiles:
                    br_img = BrightfieldTile.from_numpy(tile.img, stats=STATS)
                    br_batch = tensor(br_img).reshape(1, *tensor(br_img).shape).cuda()
                    pred_fs = unet_model(br_batch)[0].detach().cpu()
                    np_fs_img = FluorescenceTile(pred_fs).to_numpy(stats=STATS)

                    # create a SlideTile
                    fs_tile = tile
                    fs_tile.img = np_fs_img
                    fluorescence_tiles.append(fs_tile)

                fluorescence_slide = FluorescenceSlide.fromTiles(fluorescence_tiles, name=fluorescence_tiles[0].name)
                fluorescence_slides.append(fluorescence_slide)
                
            fluorescence_slides_list.append(fluorescence_slides)
        
        inference_end_time = time.time()
        if not disable_prints:
            sys.stdout.flush()
            print("")
            print("~ "*30)
            inference_time = inference_end_time-inference_start_time
            print(f"Number of slides processed {len(unique_slide_names)}")
            print(f"Inference finished in {inference_time} seconds")
            print(f"Average inference time for one slide {inference_time/len(unique_slide_names)} seconds)")
            print("~ "*30)
            print("")
            sys.stdout.flush()
            
        return fluorescence_slides_list

        
    return partial(inference_func, model_dir)
    
    
    
    
    