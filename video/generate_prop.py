import torch
from typing import Dict, List, Union
import numpy as np
import torch.nn.functional as F

class GenerateProposal():
    def Generate(prop_model: torch.nn.Module, feature_paths: Dict[str, str], pad_idx: int, cfg, device: int, duration_in_secs: float) -> torch.Tensor:
        '''Generates proposals using the pre-trained proposal model.
        Args:
            prop_model (torch.nn.Module): Pre-trained proposal model
            feature_paths (Dict): dict with paths to features ('audio', 'rgb', 'flow')
            pad_idx (int): A special padding token from train dataset.
            cfg (Config): config object used to train the proposal model
            device (int): GPU id
            duration_in_secs (float): duration of the video in seconds. Try this tool to obtain the duration:
                `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 in.mp4`

        Returns:
            torch.Tensor: tensor of size (batch=1, num_props, 3) with predicted proposals.
        '''
        # load features
        feature_stacks = load_features_from_npy(
            feature_paths, None, None, duration_in_secs, pad_idx, device, get_full_feat=True, 
            pad_feats_up_to=cfg.pad_feats_up_to
        )

        # form input batch
        batch = {
            'feature_stacks': feature_stacks,
            'duration_in_secs': duration_in_secs
        }

        with torch.no_grad():
            # masking out padding in the input features
            masks = make_masks(batch['feature_stacks'], None, cfg.modality, pad_idx)
            # inference call
            predictions, _, _, _ = prop_model(batch['feature_stacks'], None, masks)
            # (center, length) -> (start, end)
            predictions = get_corner_coords(predictions)
            # sanity-preserving clipping of the start & end points of a segment
            predictions = trim_proposals(predictions, batch['duration_in_secs'])
            # fildering out segments which has 0 or too short length (<0.2) to be a proposal
            predictions = remove_very_short_segments(predictions, shortest_segment_prior=0.2)
            # seƒlect top-[max_prop_per_vid] predictions
            predictions = select_topk_predictions(predictions, k=cfg.max_prop_per_vid)

        return predictions

def load_features_from_npy(feature_paths: Dict[str, str], start: float, end: float, duration: float, pad_idx: int, device: int, get_full_feat=False, pad_feats_up_to: Dict[str, int] = None) -> Dict[str, torch.Tensor]:
    '''Loads the pre-extracted features from numpy files. 
    This function is conceptually close to `datasets.load_feature.load_features_from_npy` but cleaned up 
    for demonstration purpose.

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
    print("pathhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
    feature_paths['audio'] = "/home/fadybassel/videobite/public/"+feature_paths['audio']
    feature_paths['rgb'] = "/home/fadybassel/videobite/public/"+feature_paths['rgb']
    feature_paths['flow'] = "/home/fadybassel/videobite/public/"+feature_paths['flow']
    stack_vggish = np.load(feature_paths['audio'] )
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
    stack_vggish = stack_vggish.to(torch.device("cpu")).unsqueeze(0)
    stack_rgb = stack_rgb.to(torch.device("cpu")).unsqueeze(0)
    stack_flow = stack_flow.to(torch.device("cpu")).unsqueeze(0)

    return {'audio': stack_vggish,'rgb': stack_rgb,'flow': stack_flow}

def make_masks(feature_stacks, captions, modality, pad_idx):
    masks = {}

    if modality == 'video':
        if captions is None:
            masks['V_mask'] = mask(feature_stacks['rgb'][:, :, 0], None, pad_idx)
        else:
            masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
    elif modality == 'audio':
        assert len(feature_stacks['audio'].shape) == 3
        if captions is None:
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
        else:
            masks['A_mask'], masks['C_mask'] = mask(feature_stacks['audio'][:, :, 0], captions, pad_idx)
    elif modality == 'audio_video':
        assert len(feature_stacks['audio'].shape) == 3
        if captions is None:
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
            masks['V_mask'] = mask(feature_stacks['rgb'][:, :, 0], None, pad_idx)
        else:
            masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
    elif modality == 'subs_audio_video':
        assert len(feature_stacks['audio'].shape) == 3
        masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
        masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
        masks['S_mask'] = mask(feature_stacks['subs'], None, pad_idx)

    return masks

def get_corner_coords(predictions):
    '''predictions (B, S*A, num_feats)'''
    starts = predictions[:, :, 0] - predictions[:, :, 1] / 2
    ends = predictions[:, :, 0] + predictions[:, :, 1] / 2
    predictions[:, :, 0] = starts
    predictions[:, :, 1] = ends
    return predictions

def trim_proposals(model_output, duration_in_secs):
    '''Changes in-place model_output (B, AS, num_feats), starts & ends are in seconds'''
    # for broadcasting it for batches
    duration_in_secs = torch.tensor(duration_in_secs, device=model_output.device).view(-1, 1)
    min_start = torch.tensor([0.0], device=model_output.device)
    # clip start for negative values and if start is longer than the duration
    model_output[:, :, 0] = model_output[:, :, 0].max(min_start).min(duration_in_secs)
    # clip end
    model_output[:, :, 1] = model_output[:, :, 1].min(duration_in_secs)
    return model_output

def remove_very_short_segments(model_output, shortest_segment_prior):
    model_output = model_output
    # (1, A*S) <-
    lengths = model_output[:, :, 1] - model_output[:, :, 0]
    # (A*S) <-
    lengths.squeeze_()
    # (A*S)
    model_output = model_output[:, lengths > shortest_segment_prior, :]

    return model_output

def select_topk_predictions(model_output, k):
    '''model_output (B, S*A, num_feats)'''
    B, S, num_feats = model_output.shape
    # sort model_output on confidence score (2nd col) within each batch
    # (B, S) <-
    indices = model_output[:, :, 2].argsort(descending=True)
    # (B, S, 1) <- .view()
    # (B, S, num_feats) <- .repeat()
    indices = indices.view(B, S, 1).repeat(1, 1, num_feats)
    model_output = model_output.gather(1, indices)
    # select top k
    # (B, k, num_feats) <-
    model_output = model_output[:, :k, :]
    return model_output

def pad_segment(feature, max_feature_len, pad_idx):
    S, D = feature.shape
    assert S <= max_feature_len
    # pad
    l, r, t, b = 0, 0, 0, max_feature_len - S
    feature = F.pad(feature, [l, r, t, b], value=pad_idx)
    return feature

def mask(src, trg, pad_idx):
    # masking the padding. src shape: (B, S') -> (B, 1, S')
    src_mask = (src != pad_idx).unsqueeze(1)
    if trg is not None:
        trg_mask = (trg != pad_idx).unsqueeze(-2) & subsequent_mask(trg.size(-1)).type_as(src_mask.data)
        return src_mask, trg_mask
    else:
        return src_mask

