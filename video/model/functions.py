import numpy as np
import torch.nn.functional as F
from typing import Dict
import torch

def load_features_from_npy(
        feature_paths: Dict[str, str], start: float, end: float, duration: float, pad_idx: int, 
        device: int, get_full_feat=False, pad_feats_up_to: Dict[str, int] = None
    ) -> Dict[str, torch.Tensor]:
    '''Loads the pre-extracted features from numpy files. 

    Args:
        feature_paths (Dict[str, str]): Paths to the numpy files (keys: 'audio', 'rgb', 'flow).
        start (float, None): Start point (in secs) of a proposal, if used for captioning the proposals.
        end (float, None): Ending point (in secs) of a proposal, if used for captioning the proposals.
        duration (float): Duration of the original video in seconds.
        pad_idx (int): The index of the padding token in the training vocabulary.
        device (int): GPU id.
        get_full_feat (bool, optional): Whether to output full, untrimmed, feature stacks. Defaults to False.
        pad_feats_up_to (Dict[str, int], optional): If get_full_feat, pad to this value. Different for audio
                                                    and video modalities. Defaults to None.

    Returns:
        Dict[str, torch.Tensor]: A dict holding 'audio', 'rgb' and 'flow' features.
    '''

    # load features. Please see README in the root folder for info on video features extraction
    stack_vggish = np.load(feature_paths['audio'])
    stack_rgb = np.load(feature_paths['rgb'])
    stack_flow = np.load(feature_paths['flow'])

    stack_vggish = torch.from_numpy(stack_vggish).float()
    stack_rgb = torch.from_numpy(stack_rgb).float()
    stack_flow = torch.from_numpy(stack_flow).float()
    
    # for proposal generation we pad the features
    if get_full_feat:
        stack_vggish = pad_segment(stack_vggish, pad_feats_up_to['audio'], pad_idx)
        stack_rgb = pad_segment(stack_rgb, pad_feats_up_to['video'], pad_idx)
        stack_flow = pad_segment(stack_flow, pad_feats_up_to['video'], pad_idx=0)
    # for captioning use trim the segment corresponding to a prop
    else:
        stack_vggish = crop_a_segment(stack_vggish, start, end, duration)
        stack_rgb = crop_a_segment(stack_rgb, start, end, duration)
        stack_flow = crop_a_segment(stack_flow, start, end, duration)

    # add batch dimension, send to device
    stack_vggish = stack_vggish.to(torch.device(device)).unsqueeze(0)
    stack_rgb = stack_rgb.to(torch.device(device)).unsqueeze(0)
    stack_flow = stack_flow.to(torch.device(device)).unsqueeze(0)

    return {'audio': stack_vggish,'rgb': stack_rgb,'flow': stack_flow}


def mask(src, trg, pad_idx):
    # masking the padding. src shape: (B, S') -> (B, 1, S')
    src_mask = (src != pad_idx).unsqueeze(1)
    if trg is not None:
        trg_mask = (trg != pad_idx).unsqueeze(-2) & subsequent_mask(trg.size(-1)).type_as(src_mask.data)
        return src_mask, trg_mask
    else:
        return src_mask

def make_masks(feature_stacks, captions, modality, pad_idx):
    masks = {}

    if captions is None:
        masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
        masks['V_mask'] = mask(feature_stacks['rgb'][:, :, 0], None, pad_idx)
    else:
        masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
        masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)

    return masks

def pad_segment(feature, max_feature_len, pad_idx):
    S, D = feature.shape
    assert S <= max_feature_len
    # pad
    l, r, t, b = 0, 0, 0, max_feature_len - S
    feature = F.pad(feature, [l, r, t, b], value=pad_idx)
    return feature


def crop_a_segment(feature, start, end, duration):
    S, D = feature.shape
    start_quantile = start / duration
    end_quantile = end / duration
    start_idx = int(S * start_quantile)
    end_idx = int(S * end_quantile)
    # handles the case when a segment is too small
    if start_idx == end_idx:
        # if the small segment occurs in the end of a video
        # [S:S] -> [S-1:S]
        if start_idx == S:
            start_idx -= 1
        # [S:S] -> [S:S+1]
        else:
            end_idx += 1
    feature = feature[start_idx:end_idx, :]

    if len(feature) == 0:
        return None
    else:
        return feature

def subsequent_mask(size):
    '''
    in: size
    out: (1, size, size)
    '''
    mask = torch.ones(1, size, size)
    mask = torch.tril(mask, 0)

    return mask.byte()