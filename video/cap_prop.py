from typing import Dict, List, Union
import torch
from video.model.functions import load_features_from_npy, make_masks


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