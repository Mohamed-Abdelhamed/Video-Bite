from typing import Dict, List, Union
import torch
import numpy as np

class CaptionProposals():
    def caption(
        cap_model: torch.nn.Module, feature_paths: Dict[str, str], 
        train_dataset: torch.utils.data.dataset.Dataset, cfg, device: int, proposals: torch.Tensor, 
        duration_in_secs: float
        ) -> List[Dict[str, Union[float, str]]]:

        '''Captions the proposals using the pre-trained model. You must specify the duration of the orignal video.

        Args:
        cap_model (torch.nn.Module): pre-trained caption model. Use load_cap_model() functions to obtain it.
        feature_paths (Dict[str, str]): dict with paths to features ('audio', 'rgb' and 'flow').
        train_dataset (torch.utils.data.dataset.Dataset): train dataset which is used as a vocab and for 
                                                            specfial tokens.
        cfg (Config): config object which was used to train caption model. pre-trained model checkpoint has it
        device (int): GPU id to calculate on.
        proposals (torch.Tensor): tensor of size (batch=1, num_props, 3) with predicted proposals.
        duration_in_secs (float): duration of the video in seconds. Try this tool to obtain the duration:
            `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 in.mp4`

        Returns:
        List(Dict(str, Union(float, str))): A list of dicts where the keys are 'start', 'end', and 'sentence'.
        '''

        results = []

        with torch.no_grad():
            for start, end, conf in proposals.squeeze():
                # load features
                feature_stacks = load_features_from_npy(
                    feature_paths, start, end, duration_in_secs, train_dataset.pad_idx, device
                )

                # decode a caption for each segment one-by-one caption word
                ints_stack = greedy_decoder(
                    cap_model, feature_stacks, cfg.max_len, train_dataset.start_idx, train_dataset.end_idx, 
                    train_dataset.pad_idx, cfg.modality
                )
                assert len(ints_stack) == 1, 'the func was cleaned to support only batch=1 (validation_1by1_loop)'

                # transform integers into strings
                strings = [train_dataset.train_vocab.itos[i] for i in ints_stack[0].cpu().numpy()]

                # remove starting token
                strings = strings[1:]
                # and remove everything after ending token
                # sometimes it is not in the list (when the caption is intended to be larger than cfg.max_len)
                try:
                    first_entry_of_eos = strings.index('</s>')
                    strings = strings[:first_entry_of_eos]
                except ValueError:
                    pass

                # join everything together
                sentence = ' '.join(strings)
                # Capitalize the sentence
                sentence = sentence.capitalize()

                # add results to the list
                results.append({
                    'start': round(start.item(), 1), 
                    'end': round(end.item(), 1), 
                    'sentence': sentence
                })        

        return results

def load_features_from_npy(
        feature_paths: Dict[str, str], start: float, end: float, duration: float, pad_idx: int, 
        device: int, get_full_feat=False, pad_feats_up_to: Dict[str, int] = None
    ) -> Dict[str, torch.Tensor]:
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

def greedy_decoder(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality):
    assert model.training is False, 'call model.eval first'

    with torch.no_grad():
        
        if 'audio' in modality:
            B, _Sa_, _Da_ = feature_stacks['audio'].shape
            device = feature_stacks['audio'].device
        elif modality == 'video':
            B, _Sv_, _Drgb_ = feature_stacks['rgb'].shape
            device = feature_stacks['rgb'].device
        else:
            raise Exception(f'Unknown modality: {modality}')

        # a mask containing 1s if the ending tok occured, 0s otherwise
        # we are going to stop if ending token occured in every sequence
        completeness_mask = torch.zeros(B, 1).byte().to(device)
        trg = (torch.ones(B, 1) * start_idx).long().to(device)

        while (trg.size(-1) <= max_len) and (not completeness_mask.all()):
            masks = make_masks(feature_stacks, trg, modality, pad_idx)
            preds = model(feature_stacks, trg, masks)
            next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1)
            trg = torch.cat([trg, next_word], dim=-1)
            completeness_mask = completeness_mask | torch.eq(next_word, end_idx).byte()

        return trg

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
    

def pad_segment(feature, max_feature_len, pad_idx):
    S, D = feature.shape
    assert S <= max_feature_len
    # pad
    l, r, t, b = 0, 0, 0, max_feature_len - S
    feature = F.pad(feature, [l, r, t, b], value=pad_idx)
    return feature

def subsequent_mask(size):
    '''
    in: size
    out: (1, size, size)
    '''
    mask = torch.ones(1, size, size)
    mask = torch.tril(mask, 0)

    return mask.byte()