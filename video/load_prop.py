from video.model.blocks import FeatureEmbedder, Identity, PositionalEncoder, VocabularyEmbedder #THIS
from video.model.encoders import BiModalEncoder #THIS
from video.model.decoders import BiModelDecoder #THIS
from pprint import pprint
import torch
import torch.nn as nn

class LoadPropModel():
    def load(self, device, prop_generator_model_path, pretrained_cap_model_path, max_prop_per_vid) -> tuple:
        '''Loading pre-trained proposal generator and config object which was used to train the model.
        Args:
            device (int): GPU id.
            prop_generator_model_path (str): Path to the pre-trained proposal generation model.
            pretrained_cap_model_path (str): Path to the pre-trained captioning module (prop generator uses the 
                                            encoder weights).
            max_prop_per_vid (int): Maximum number of proposals per video.

        Returns:
            Config, torch.nn.Module: config, proposal generator
        '''
        # load and patch the config for user-defined arguments
        checkpoint = torch.load(prop_generator_model_path, map_location='cpu')
        cfg = checkpoint['config']
        print("cfgggggggggggggggggggggggggggg")
        # cfg.device = device ####################################################################################4
        cfg.device = device
        cfg.max_prop_per_vid = max_prop_per_vid
        cfg.pretrained_cap_model_path = pretrained_cap_model_path
        cfg.train_meta_path = './video/data/train.csv'  # in the saved config it is named differently

        # load anchors
        anchors = {
            'audio': checkpoint['anchors']['audio'],
            'video': checkpoint['anchors']['video']
        }

        # define model and load the weights
        model = MultimodalProposalGenerator(cfg, anchors)
        # device = torch.device(cfg.device) ###############################################################################6

        # torch.cuda.set_device(device)
        # torch.device('cpu',0)
        torch.device("cpu")
        device = torch.device("cpu")
        cfg.device=device
        #torch.set_device(device) ####################################################################################35
        model.load_state_dict(checkpoint['model_state_dict'])  # if IncompatibleKeys - ignore
        model = model.to(cfg.device)
        model.eval()

        return cfg, model

class MultimodalProposalGenerator(nn.Module):

    def __init__(self, cfg, anchors):
        super(MultimodalProposalGenerator, self).__init__()
        assert cfg.modality == 'audio_video'
        self.cfg = cfg
        self.anchors = anchors
        self.EPS = 1e-16
        self.num_logits = 3  # 3: c, w, obj
        
        self.emb_V = Identity()
        self.emb_A = Identity()
        self.pos_enc_V = PositionalEncoder(cfg.d_model_video, cfg.dout_p)
        self.pos_enc_A = PositionalEncoder(cfg.d_model_audio, cfg.dout_p)

        # load the pre-trained encoder from captioning module
       
        print(f'Pretrained caption path: \n {cfg.pretrained_cap_model_path}')
        print(self.cfg.device)
        cap_model_cpt = torch.load(cfg.pretrained_cap_model_path, map_location='cpu')
        encoder_config = cap_model_cpt['config']
        self.encoder = BiModalEncoder(
            encoder_config.d_model_audio, encoder_config.d_model_video, encoder_config.d_model, 
            encoder_config.dout_p, encoder_config.H, encoder_config.d_ff_audio, 
            encoder_config.d_ff_video, encoder_config.N
        )
        encoder_weights = {k: v for k, v in cap_model_cpt['model_state_dict'].items() if 'encoder' in k}
        encoder_weights = {k.replace('module.encoder.', ''): v for k, v in encoder_weights.items()}
        self.encoder.load_state_dict(encoder_weights)
        self.encoder = self.encoder.to(torch.device("cpu"))
        for param in self.encoder.parameters():
            param.requires_grad = cfg.finetune_cap_encoder
        
        dims_A = [cfg.d_model_audio, *cfg.conv_layers_audio, self.num_logits*cfg.anchors_num_audio]
        dims_V = [cfg.d_model_video, *cfg.conv_layers_video, self.num_logits*cfg.anchors_num_video]
        self.detection_layers_A = torch.nn.ModuleList([
            ProposalGenerationHead(dims_A, k, cfg.dout_p, cfg.layer_norm) for k in cfg.kernel_sizes['audio']
        ])
        self.detection_layers_V = torch.nn.ModuleList([
            ProposalGenerationHead(dims_V, k, cfg.dout_p, cfg.layer_norm) for k in cfg.kernel_sizes['video']
        ])
        
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward_modality(self, x, targets, detection, stride, anchors_list):
        anchors_num = len(anchors_list)
        # in case targets is None
        loss = 0
        losses = {}

        x = detection(x)

        B, S, D = x.shape
        x = x.view(B, S, anchors_num, self.num_logits)

        x = x.permute(0, 2, 1, 3).contiguous()
        grid_cell = torch.arange(S).view(1, 1, S).float().to(self.cfg.device)
        # After dividing anchors by the stride, they represent the size size of
        # how many grid celts they are overlapping: 1.2 = 1 and 20% of a grid cell.
        # After multiplying them by the stride, the pixel values are going to be
        # obtained.
        anchors_list = [[anchor / stride] for anchor in anchors_list]
        anchors_tensor = torch.tensor(anchors_list, device="cpu")
        # (A, 2) -> (1, A, 1) for broadcasting
        prior_length = anchors_tensor.view(1, anchors_num, 1)

        # prediction values for the *loss* calculation (training)
        sigma_c = torch.sigmoid(x[:, :, :, 0])  # center
        l = x[:, :, :, 1]  # length
        sigma_o = torch.sigmoid(x[:, :, :, 2])  # objectness

        # prediction values that are going to be used for the original image
        # we need to detach them from the graph as we don't need to backproparate
        # on them
        predictions = x.clone().detach()
        # broadcasting (B, A, S) + (1, 1, S)
        # For now, we are not going to multiply them by stride since
        # we need them in make_targets
        predictions[:, :, :, 0] = sigma_c + grid_cell
        # broadcasting (1, A, 1) * (B, A, S)
        predictions[:, :, :, 1] = prior_length * torch.exp(l)
        predictions[:, :, :, 2] = sigma_o

        if targets is not None:
            obj_mask, noobj_mask, gt_x, gt_w, gt_obj = make_targets(predictions, targets, 
                                                                    anchors_tensor, stride)
            ## Loss
            # Localization
            loss_x = self.mse_loss(sigma_c[obj_mask], gt_x[obj_mask])
            loss_w = self.mse_loss(l[obj_mask], gt_w[obj_mask])
            loss_loc = loss_x + loss_w
            # Confidence
            loss_obj = self.bce_loss(sigma_o[obj_mask], gt_obj[obj_mask])
            loss_noobj = self.bce_loss(sigma_o[noobj_mask], gt_obj[noobj_mask])
            loss_conf = self.cfg.obj_coeff * loss_obj + self.cfg.noobj_coeff * loss_noobj
            # Total loss
            loss = loss_loc + loss_conf

            losses = {
                'loss_x': loss_x,
                'loss_w': loss_w,
                'loss_conf_obj': loss_obj,
                'loss_conf_noobj': loss_noobj
            }

        # for NMS: (B, A, S, 3) -> (B, A*S, 3)
        predictions = predictions.view(B, S*anchors_num, self.num_logits)
        predictions[:, :, :2] *= stride

        return predictions, loss, losses

    def forward(self, x, targets, masks):
        V, A = x['rgb'] + x['flow'], x['audio']

        # (B, Sm, Dm) < - (B, Sm, Dm), m in [a, v]
        A = self.emb_A(A)
        V = self.emb_V(V)
        A = self.pos_enc_A(A)
        V = self.pos_enc_V(V)
        # notation: M1m2m2 (B, Sm1, Dm1), M1 is the target modality, m2 is the source modality
        Av, Va = self.encoder((A, V), masks)

        all_predictions_A = []
        all_predictions_V = []
        # total_loss should have backward
        sum_losses_dict_A = {}
        sum_losses_dict_V = {}
        total_loss_A = 0
        total_loss_V = 0

        for layer in self.detection_layers_A:
            props_A, loss_A, losses_A = self.forward_modality(
                Av, targets, layer, self.cfg.strides['audio'], self.anchors['audio']
            )
            total_loss_A += loss_A
            all_predictions_A.append(props_A)
            sum_losses_dict_A = add_dict_to_another_dict(losses_A, sum_losses_dict_A)

        for layer in self.detection_layers_V:
            props_V, loss_V, losses_V = self.forward_modality(
                Va, targets, layer, self.cfg.strides['video'], self.anchors['video']
            )
            total_loss_V += loss_V
            all_predictions_V.append(props_V)
            sum_losses_dict_V = add_dict_to_another_dict(losses_V, sum_losses_dict_V)

        all_predictions_A = torch.cat(all_predictions_A, dim=1)
        all_predictions_V = torch.cat(all_predictions_V, dim=1)

        total_loss = total_loss_A + total_loss_V

        # combine predictions
        all_predictions = torch.cat([all_predictions_A, all_predictions_V], dim=1)
        # if you like the predictions to be half from audio and half from the video modalities
        # all_predictions = torch.cat([
        #     select_topk_predictions(all_predictions_A, k=self.cfg.max_prop_per_vid // 2),
        #     select_topk_predictions(all_predictions_V, k=self.cfg.max_prop_per_vid // 2)
        # ], dim=1)

        return all_predictions, total_loss, sum_losses_dict_A, sum_losses_dict_V

class ProposalGenerationHead(nn.Module):

    def __init__(self, d_model_list, kernel_size, dout_p, layer_norm=False):
        super(ProposalGenerationHead, self).__init__()
        assert kernel_size % 2 == 1, 'It is more convenient to use odd kernel_sizes for padding'
        conv_layers = []
        in_dims = d_model_list[:-1]
        out_dims = d_model_list[1:]
        N_layers = len(d_model_list) - 1

        for n, (in_d, out_d) in enumerate(zip(in_dims, out_dims)):
            if layer_norm:
                conv_layers.append(Transpose())
                conv_layers.append(nn.LayerNorm(in_d))
                conv_layers.append(Transpose())

            if n == 0:
                conv_layers.append(nn.Conv1d(in_d, out_d, kernel_size, padding=kernel_size//2))
            else:
                conv_layers.append(nn.Conv1d(in_d, out_d, kernel_size=1))

            if n < (N_layers - 1):
                if dout_p > 0:
                    conv_layers.append(nn.Dropout(dout_p))
                conv_layers.append(nn.ReLU())

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        # (B, D, S) <- (B, S, D)
        x = x.permute(0, 2, 1)
        # (B, d, S) <- (B, D, S)
        x = self.conv_layers(x)
        # (B, S, d) <- (B, d, S)
        x = x.permute(0, 2, 1)
        # x = self.fc_layer(x)
        return x


def make_targets(predictions, targets, anchors, stride):
    '''
        The implementation relies on YOLOv3 for object detection
            - https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py
            - https://github.com/v-iashin/PersonalProjects/blob/master/detector/darknet.py
    '''
    B, num_anchs, G, num_feats = predictions.size()

    # classes = 1
    EPS = 1e-16

    # create the placeholders
    noobj_mask = torch.ones(B, num_anchs, G,device="cpu" ).bool() #########################################################################3
    obj_mask = torch.zeros_like(noobj_mask).bool()
    target_x = torch.zeros_like(noobj_mask).float()
    target_w = torch.zeros_like(noobj_mask).float()

    # image index within the batch, the g.t. label of an object on the image
    vid_idx = targets[:, 0].long()
    # ground truth center coordinates and bbox dimensions
    # since the target bbox coordinates are in seconds, we transoform
    # them into grid-axis
    # So, the gt_x will represent the position of the center in grid cells
    # Similarly, the size sizes are also scaled to grid size
    gt_x = targets[:, 1] / stride
    gt_w = targets[:, 2] / stride
    # ious between scaled anchors (anchors_from_cfg / stride) and gt bboxes
    gt_anchor_ious = tiou_vectorized(anchors, gt_w.unsqueeze(-1), without_center_coords=True)
    # selecting the best anchors for the g.t. bboxes
    best_ious, best_anchors = gt_anchor_ious.max(dim=0)

    # remove a decimal part -> gi point to the grid position to which an object will correspond 
    # for example: 9.89 -> 9
    gt_cell = gt_x.long()
    # helps with RuntimeError: CUDA error: device-side assert triggered
    # This aims to avoid gt_cell[i] exceeding the bounds
    gt_cell[gt_cell < 0] = 0
    gt_cell[gt_cell > G - 1] = G - 1
    # update the obj and noobj masks.
    # the noobj mask has 0 where obj mask has 1 and where IoU between
    # g.t. bbox and anchor is higher than ignore_thresh
    obj_mask[vid_idx, best_anchors, gt_cell] = 1
    noobj_mask[vid_idx, best_anchors, gt_cell] = 0

    # center shift: for example 9.89 -> 0.89
    target_x[vid_idx, best_anchors, gt_cell] = gt_x - gt_x.floor()

    # Yolo predicts the coefficients (log of coefs actually, see exp() in the paper) 
    # that will be used to multiply with anchor lengths for predicted segment lengths
    # Therefore, we modify targets and apply log transformation.
    target_w[vid_idx, best_anchors, gt_cell] = torch.log(gt_w.t() / anchors[best_anchors][:, 0] + EPS)
    # since YOLO loss penalizes the model only for wrong predictions at the ground truth
    # cells, we extract only these predictions from the tensor
    # Extracting the labels from the prediciton tensor
    pred_x_w = predictions[vid_idx, best_anchors, gt_cell, :2]

    # ground truth objectness
    target_obj = obj_mask.float()

    return obj_mask, noobj_mask, target_x, target_w, target_obj

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


def add_dict_to_another_dict(one_dict, another_dict):
    another_dict = {k: another_dict.get(k, 0) + v for k, v in one_dict.items()}
    return another_dict