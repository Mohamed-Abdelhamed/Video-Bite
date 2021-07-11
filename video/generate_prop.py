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
        nms_tiou_thresh = 0.4
        # load features
        feature_stacks = load_features_from_npy(
            feature_paths, None, None, duration_in_secs, pad_idx, device, get_full_feat=True, 
            pad_feats_up_to=cfg.pad_feats_up_to
        )

        with torch.no_grad():
            # masking out padding in the input features
            masks = make_masks(feature_stacks, None, cfg.modality, pad_idx)
            # inference call
            predictions, _, _, _ = prop_model(feature_stacks, None, masks)
            # (center, length) -> (start, end)
            predictions = get_corner_coords(predictions)
            # sanity-preserving clipping of the start & end points of a segment
            predictions = trim_proposals(predictions, duration_in_secs)
            # fildering out segments which has 0 or too short length (<0.2) to be a proposal
            predictions = remove_very_short_segments(predictions, shortest_segment_prior=0.2)
            # seƒlect top-[max_prop_per_vid] predictions
            predictions = select_topk_predictions(predictions, k=cfg.max_prop_per_vid)

            predictions = non_max_suppresion(predictions.squeeze(), nms_tiou_thresh)
            predictions = predictions.unsqueeze(0)

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

def tiou_vectorized(segments1, segments2):

    M, D = segments1.shape
    N, D = segments2.shape

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
        tious = tiou_vectorized(video_preds[0, :].unsqueeze(0), video_preds[1:, :])
        # (*) <- (1, *)
        tious = tious.reshape(-1)
        # (*', num_feats)
        video_preds = video_preds[1:, :][tious < tIoU_threshold]
    # (new_N, D) <- a list of (1, num_feats)
    model_output = torch.cat(model_output_after_nms)
    return model_output