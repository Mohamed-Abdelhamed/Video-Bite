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

def tiou_vectorized(segments1, segments2, without_center_coords=False, center_length=True):

    def center_length_2_start_end(segments):
        '''there is get_corner_coords(predictions) and has a bit diffrenrent logic. both are kept'''
        start = segments[:, 0] - segments[:, 1] / 2
        end = segments[:, 0] + segments[:, 1] / 2
        return start, end

    # add 'fake' center coordinates. You can use any value, we use zeros
    if without_center_coords:
        segments1 = torch.cat([torch.zeros_like(segments1), segments1], dim=1)
        segments2 = torch.cat([torch.zeros_like(segments2), segments2], dim=1)

    M, D = segments1.shape
    N, D = segments2.shape

    # TODO: replace with get_corner_coords from localization_utils
    if center_length:
        start1, end1 = center_length_2_start_end(segments1)
        start2, end2 = center_length_2_start_end(segments2)
    else:
        start1, end1 = segments1[:, 0], segments1[:, 1]
        start2, end2 = segments2[:, 0], segments2[:, 1]

    # broadcasting
    start1 = start1.view(M, 1)
    end1 = end1.view(M, 1)
    start2 = start2.view(1, N)
    end2 = end2.view(1, N)

    # calculate segments for intersection
    intersection_start = torch.max(start1, start2)
    intersection_end = torch.min(end1, end2)

    # we make sure that the area is 0 if size of a side is negative
    # which means that intersection_start > intersection_end which is not feasible
    # Note: adding one because the coordinates starts at 0 and let's
    intersection = torch.clamp(intersection_end - intersection_start, min=0.0)

    # finally we calculate union for each pair of segments
    union1 = (end1 - start1)
    union2 = (end2 - start2)
    union = union1 + union2 - intersection
    union = torch.min(torch.max(end1, end2) - torch.min(start1, start2), union)

    tious = intersection / (union + 1e-8)
    return tious

def non_max_suppresion(video_preds, tIoU_threshold):
    '''video_preds (AS, num_features)'''
    # model_output should be sorted according to conf_score, otherwise sort it here
    model_output_after_nms = []
    while len(video_preds) > 0:
        # (1, num_feats) <- (one_vid_pred[0, :].unsqueeze(0))
        model_output_after_nms.append(video_preds[0, :].unsqueeze(0))
        if len(video_preds) == 1:
            break
        # (1, *) <- (1, num_feats) x (*, num_feats)
        tious = tiou_vectorized(video_preds[0, :].unsqueeze(0), video_preds[1:, :], 
                                center_length=False)
        # (*) <- (1, *)
        tious = tious.reshape(-1)
        # (*', num_feats)
        video_preds = video_preds[1:, :][tious < tIoU_threshold]
    # (new_N, D) <- a list of (1, num_feats)
    model_output = torch.cat(model_output_after_nms)
    return model_output

class VideoToText:
    def __init__(self,video):
        self.video = video
        # self.video = "/home/markrefaat/videobite/public/skating"
    
    def extractText(self,features):
        ## parser = argparse.ArgumentParser(description='One video prediction')
        ## parser.add_argument('--video_name', required=True)
        ## args = parser.parse_args()
        print("VIDEDDDDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        print(self.video)

        print("VIDEDDDDEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        print(features)

        prop_generator_model_path = PROP_GENERATOR_MODEL_PATH
        pretrained_cap_model_path = PRETRAINED_CAP_MODEL_PATH
        
        clip = VideoFileClip(self.video)
      
        duration_in_secs = clip.duration
        device_id = 0
        max_prop_per_vid = 100
        nms_tiou_thresh = 0.4

        feature_paths = {
            'audio': DEFAULT_PATH + features + "-vggish.npy",
            'rgb': DEFAULT_PATH + features + "-rgb.npy",
            'flow': DEFAULT_PATH + features +"-flow.npy",
        }
        # Loading models and other essential stuff
        cap_cfg, cap_model, train_dataset = LoadCapModel().load(pretrained_cap_model_path, device_id)
        prop_cfg, prop_model = LoadPropModel().load(
            device_id, prop_generator_model_path, pretrained_cap_model_path, max_prop_per_vid
        )
        
        # Proposal
        proposals =  GenerateProposal.Generate(
            prop_model, feature_paths, train_dataset.pad_idx, prop_cfg, device_id, duration_in_secs
        )
        # NMS if specified
        if nms_tiou_thresh is not None:
            proposals = non_max_suppresion(proposals.squeeze(), nms_tiou_thresh)
            proposals = proposals.unsqueeze(0)

        captions = CaptionProposals.caption(
            cap_model, feature_paths, train_dataset, cap_cfg, device_id, proposals, duration_in_secs
        )

        captions = sorted(captions, key=lambda k: k['start'])
        print(captions)
        return captions
        # return [{'start': 0.0, 'end': 202.1, 'sentence': 'A boy is skateboarding on a ramp'}, {'start': 0.0, 'end': 23.4, 'sentence': 'A person is seen riding a skateboard down a hill and performing tricks on a ramp'}, {'start': 0.0, 'end': 7.3, 'sentence': 'A man is seen sitting on a beach with a camera and leads into a man riding on a surfboard on a beach'}, {'start': 43.6, 'end': 46.4, 'sentence': 'A man is seen walking down a street with a camera'}, {'start': 46.2, 'end': 49.0, 'sentence': 'A man is seen walking down a ramp'}, {'start': 41.1, 'end': 43.8, 'sentence': 'A man is seen walking down a ramp while a man is shown on a skateboard'}, {'start': 49.0, 'end': 51.6, 'sentence': 'A man is seen walking down a street'}, {'start': 194.3, 'end': 202.0, 'sentence': 'The video ends with the closing credits'}, {'start': 27.2, 'end': 58.7, 'sentence': 'A man is skateboarding on a ramp'}]
