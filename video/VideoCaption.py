import argparse
# from constants import DEFAULT_PATH, PRETRAINED_CAP_MODEL_PATH, PROP_GENERATOR_MODEL_PATH
PROP_GENERATOR_MODEL_PATH = "./video/sample/best_prop_model.pt"
PRETRAINED_CAP_MODEL_PATH = './video/sample/best_cap_model.pt'
DEFAULT_PATH = "/home/fadybassel/videobite flask/videos/features/"
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch 
from moviepy.editor import VideoFileClip 
 
from video.load_cap import LoadCapModel 
from video.load_prop import LoadPropModel 
from video.generate_prop import GenerateProposal 
from video.cap_prop import CaptionProposals 
 
 
class VideoToText: 
    def __init__(self,video): 
        # self.video = "/home/markrefaat/videobite/public/skating" 
        self.video = video 
     
    def extractText(self, features): 
        prop_generator_model_path = PROP_GENERATOR_MODEL_PATH 
        pretrained_cap_model_path = PRETRAINED_CAP_MODEL_PATH 
        clip = VideoFileClip(self.video) 
        duration_in_secs = clip.duration 
        device_id = 0 
        max_prop_per_vid = 100 
         
 
        feature_paths = { 
            'audio': DEFAULT_PATH + features + "-vggish.npy", 
            'rgb': DEFAULT_PATH + features + "-rgb.npy", 
            'flow': DEFAULT_PATH + features +"-flow.npy", 
        } 
        # Loading models and other essential stuff 
        cap_cfg, cap_model, train_dataset, encoder_weights= LoadCapModel().load(pretrained_cap_model_path, device_id) 
         
        prop_cfg, prop_model = LoadPropModel().load( 
            device_id, prop_generator_model_path, pretrained_cap_model_path, max_prop_per_vid, cap_cfg, encoder_weights 
        ) 
        # Proposal 
        proposals =  GenerateProposal.Generate( 
            prop_model, feature_paths, train_dataset.pad_idx, prop_cfg, device_id, duration_in_secs 
        ) 
 
        captions = CaptionProposals.caption( 
            cap_model, feature_paths, train_dataset, cap_cfg, device_id, proposals, duration_in_secs 
        ) 
 
        captions = sorted(captions, key=lambda k: k['start']) 
        print(captions) 
        return captions 