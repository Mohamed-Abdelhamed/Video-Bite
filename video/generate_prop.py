import torch
from typing import Dict
from video.model.functions import load_features_from_npy, make_masks


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

