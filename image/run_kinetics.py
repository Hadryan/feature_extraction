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
from multiprocessing import Process

def extract_feats(paths, params):
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

    model = model.cuda()
    model.eval()

    for category in tqdm(paths):
        save_path = os.path.join(params['feat_path'], category)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        frames_path_list = glob.glob(os.path.join(params['frame_path'], category, '*'))

        for frames_dst in tqdm(frames_path_list):
            file_name = frames_dst.split('/')[-1] + '.pkl'
            if os.path.exists(os.path.join(save_path, file_name)):
                continue
            
            image_list = sorted(glob.glob(os.path.join(frames_dst, '*.%s' % params['frame_suffix'])))
            if len(image_list) == 0:
                continue

            if params['k'] and len(image_list) > params['k']:
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
                feats = model.features(images.cuda().to(torch.float32))
                logits = model.logits(feats)
                
            feats = feats.squeeze().mean(-1).mean(-1).cpu().numpy()
            logits = logits.squeeze().cpu().numpy()

            tqdm.write('%s: %s %s' % (file_name, str(feats.shape), str(logits.shape)))

            with open(os.path.join(save_path, file_name), 'wb') as f:
                f.write(pickle.dumps([feats, logits]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", type=str, required=True, help='the path to load all the frames')
    parser.add_argument("--feat_path", type=str, required=True, help='the path you want to save the features')

    parser.add_argument("--gpu", type=str, default='0', help='set CUDA_VISIBLE_DEVICES environment variable')
    parser.add_argument("--model", type=str, default='inceptionresnetv2', help='inceptionresnetv2 | resnet101')
    
    parser.add_argument("--k", type=int, default=60, 
        help='uniformly sample k frames from the existing frames and then extract their features. k=0 will extract all existing frames')
    parser.add_argument("--frame_suffix", type=str, default='jpg')
    parser.add_argument("--num_processes", type=int, default=3)
    
    args = parser.parse_args()
    params = vars(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

    assert os.path.exists(params['frame_path'])
    if not os.path.exists(params['feat_path']):
        os.makedirs(params['feat_path'])

    print('Model: %s' % params['model'])
    print('The extracted features will be saved to --> %s' % params['feat_path'])

    paths = os.listdir(params['frame_path'])
    paths = np.array_split(paths, params['num_processes'])
    processes = []
    for i in range(params['num_processes']):
        proc = Process(target=extract_feats, args=(paths[i], params))
        proc.start()
        processes.append(proc)

    for i in range(params['num_processes']):
        processes[i].join()
    

'''
python run_kinetics.py \
--frame_path /disk2/zhangcan/dataset/kinetics_frames \
--feat_path /home/yangbang/VideoCaptioning/kinetics/feats/image_IRv2 \
--model inceptionresnetv2 \
--k 60 \
--frame_suffix jpg \
--gpu 0 \
--num_processes 3
'''
