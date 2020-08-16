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


def extract_feats(params, model, load_image_fn, C, H, W):
    model.eval()

    frames_path_list = glob.glob(os.path.join(params['frame_path'], '*'))
    if not params['not_extract_feat']: 
        db = h5py.File(params['feat_dir'], 'a')
    if params['extract_logit']: 
        db2 = h5py.File(params['logit_dir'], 'a')
    

    for frames_dst in tqdm(frames_path_list):
        video_id = frames_dst.split('/')[-1]
        if int(video_id[5:]) > 10000: continue
        if (not params['not_extract_feat'] and video_id in db.keys()) or (params['extract_logit'] and video_id in db2.keys()):
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

        tqdm.write('%s: %s %s' % (video_id, str(feats.shape), str(logits.shape)))

        if not params['not_extract_feat']: 
            db[video_id] = feats
        if params['extract_logit']: 
            db2[video_id] = logits

    if not params['not_extract_feat']: 
        db.close()
    if params['extract_logit']:       
        db2.close()  

def test_latency(params, model, load_image_fn, C, H, W):
    assert params['test_latency'] > 0
    import time

    model.eval()
    frames_path_list = glob.glob(os.path.join(params['frame_path'], '*'))[:params['test_latency']]
    n_frames = 8
    total_time = 0
    for frames_dst in tqdm(frames_path_list):
        video_id = frames_dst.split('/')[-1]
        image_list = sorted(glob.glob(os.path.join(frames_dst, '*.%s' % params['frame_suffix'])))
        images = torch.zeros((n_frames, C, H, W))
        bound = [int(i) for i in np.linspace(0, len(image_list), n_frames+1)]
        for i in range(n_frames):
            idx = (bound[i] + bound[i+1]) // 2
            if params['model'] == 'googlenet':
                images[i] = load_image_fn.get(image_list[idx])
            else:
                images[i] = load_image_fn(image_list[idx])

        with torch.no_grad():
            start_time = time.time()
            feats = logits = model(images.cuda())
            total_time += (time.time()-start_time)
    print(total_time, total_time/params['test_latency'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", type=str, required=True, help='the path to load all the frames')
    parser.add_argument("--feat_path", type=str, required=True, help='the path you want to save the features')
    parser.add_argument("--feat_name", type=str, default='', help='the name of the features file, saved in hdf5 format')
    parser.add_argument("--logit_name", type=str, default='', help='the name of the logits file, saved in hdf5 format')
    
    parser.add_argument("-nef", "--not_extract_feat", default=False, action='store_true')
    parser.add_argument("-el", "--extract_logit", default=False, action='store_true')
    
    parser.add_argument("--gpu", type=str, default='0', help='set CUDA_VISIBLE_DEVICES environment variable')
    parser.add_argument("--model", type=str, default='inceptionresnetv2', help='inceptionresnetv2 | resnet101')
    
    parser.add_argument("--k", type=int, default=60, 
        help='uniformly sample k frames from the existing frames and then extract their features. k=0 will extract all existing frames')
    parser.add_argument("--frame_suffix", type=str, default='jpg')

    parser.add_argument("--test_latency", type=int, default=0)
    
    args = parser.parse_args()
    params = vars(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

    assert os.path.exists(params['frame_path'])
    if not os.path.exists(params['feat_path']):
        os.makedirs(params['feat_path'])

    if not params['not_extract_feat']: assert params['feat_name']
    if params['extract_logit']: assert params['logit_name']

    params['feat_dir'] = os.path.join(params['feat_path'], params['feat_name'] + ('' if '.hdf5' in params['feat_name'] else '.hdf5'))
    params['logit_dir'] = os.path.join(params['feat_path'], params['logit_name'] + ('' if '.hdf5' in params['logit_name'] else '.hdf5'))

    print('Model: %s' % params['model'])
    print('The extracted features will be saved to --> %s' % params['feat_dir'])

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
        model = pretrainedmodels.inceptionresnetv2(
            num_classes=1001, pretrained='imagenet+background')
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
    if params['test_latency']:
        test_latency(params, model, load_image_fn, C, H, W)
    else:
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