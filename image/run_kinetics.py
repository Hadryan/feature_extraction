import glob
from tqdm import tqdm
import numpy as np
import os
import argparse
import torch
import pretrainedmodels
from pretrainedmodels import utils
import h5py
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.model_zoo as model_zoo
from PIL import Image
import pickle

def extract_feats(params, model, load_image_fn, C, H, W):
    model.eval()

    for category in tqdm(os.listdir(params['frame_path'])):
        save_path = os.path.join(params['feat_path'], category)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        frames_path_list = glob.glob(os.path.join(params['frame_path'], category, '*'))

        for frames_dst in tqdm(frames_path_list):
            file_name = frames_dst.split('/')[-1] + '.pkl'
            if os.path.exists(os.path.join(save_path, file_name)):
                continue
            
            image_list = sorted(glob.glob(os.path.join(frames_dst, '*.%s' % params['frame_suffix'])))

            if params['k']: 
                images = torch.zeros((params['k'], C, H, W))
                bound = [int(i) for i in np.linspace(0, len(image_list), params['k']+1)]
                for i in range(params['k']):
                    idx = (bound[i] + bound[i+1]) // 2
                    images[i] = load_image_fn(image_list[idx])
            else:
                images = torch.zeros((len(image_list), C, H, W))
                for i, image_path in enumerate(image_list):
                    images[i] = load_image_fn(image_path)

            with torch.no_grad():
                feats = model.features(images.cuda())
                logits = model.logits(feats)
                
            feats = feats.squeeze().cpu().numpy()
            logits = logits.squeeze().cpu().numpy()

            tqdm.write('%s: %s %s' % (file_name, str(feats.shape), str(logits.shape)))

            with open(os.path.join(save_path, file_name), 'wb') as f:
                f.write(pickle.dumps({'feats': feats, 'logits': logits}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", type=str, required=True, help='the path to load all the frames')
    parser.add_argument("--feat_path", type=str, required=True, help='the path you want to save the features')

    parser.add_argument("--gpu", type=str, default='0', help='set CUDA_VISIBLE_DEVICES environment variable')
    parser.add_argument("--model", type=str, default='inceptionresnetv2', help='inceptionresnetv2 | resnet101')
    
    parser.add_argument("--k", type=int, default=60, 
        help='uniformly sample k frames from the existing frames and then extract their features. k=0 will extract all existing frames')
    parser.add_argument("--frame_suffix", type=str, default='jpg')
    
    args = parser.parse_args()
    params = vars(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

    assert os.path.exists(params['frame_path'])
    if not os.path.exists(params['feat_path']):
        os.makedirs(params['feat_path'])

    if not params['not_extract_feat']: assert params['feat_name']
    if params['extract_logit']: assert params['logit_name']

    print('Model: %s' % params['model'])
    print('The extracted features will be saved to --> %s' % params['feat_path'])

    if params['model'] == 'resnet101':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet101(pretrained='imagenet')
    elif params['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(pretrained='imagenet')
    elif params['model'] == 'resnet18':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet18(pretrained='imagenet')
    elif params['model'] == 'resnet34':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet34(pretrained='imagenet')
    elif params['model'] == 'inceptionresnetv2':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionresnetv2(pretrained='imagenet')
    elif params['model'] == 'googlenet':
        C, H, W = 3, 224, 224
        model = googlenet(pretrained=True)
        print(model)
    else:
        print("doesn't support %s" % (params['model']))

    load_image_fn = utils.LoadTransformImage(model)
    model.last_linear = utils.Identity() 

    model = model.cuda()

    #summary(model, (C, H, W))
    extract_feats(params, model, load_image_fn, C, H, W)

'''
python extract_image_feats_from_frames.py \
--frame_path "/home/yangbang/VideoCaptioning/MSRVTT/all_frames/" \
--feat_path "/home/yangbang/VideoCaptioning/MSRVTT/feats/" \
--feat_name msrvtt_R152 \
--model resnet152 \
--k 60 \
--frame_suffix jpg \
--gpu 2

python extract_image_feats_from_frames.py \
--frame_path "/home/yangbang/VideoCaptioning/Youtube2Text/all_frames/" \
--feat_path "/home/yangbang/VideoCaptioning/Youtube2Text/feats/" \
--feat_name msvd_R152 \
--model resnet152 \
--k 60 \
--frame_suffix jpg \
--gpu 3
'''