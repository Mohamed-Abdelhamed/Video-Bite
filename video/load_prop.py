from video.model.blocks import Identity, PositionalEncoder
from video.model.encoders import BiModalEncoder
from video.model.proposal_generator import ProposalGenerationHead

import torch
import torch.nn as nn

class LoadPropModel():
    def load(self, device, prop_generator_model_path, pretrained_cap_model_path, max_prop_per_vid, cap_cfg, encoder_weights) -> tuple:
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
        cfg.device = device
        cfg.max_prop_per_vid = max_prop_per_vid
        cfg.pretrained_cap_model_path = pretrained_cap_model_path

        # load anchors
        anchors = {
            'audio': checkpoint['anchors']['audio'],
            'video': checkpoint['anchors']['video']
        }

        # define model and load the weights
        model = MultimodalProposalGenerator(cfg, anchors, cap_cfg, encoder_weights)
        device = torch.device(cfg.device)
        torch.cuda.set_device(device)
        model.load_state_dict(checkpoint['model_state_dict'])  # if IncompatibleKeys - ignore
        model = model.to(cfg.device)
        model.eval()

        return cfg, model

class MultimodalProposalGenerator(nn.Module):

    def __init__(self, cfg, anchors, cap_cfg, encoder_weights):
        super(MultimodalProposalGenerator, self).__init__()
        self.cfg = cfg
        self.anchors = anchors
        self.num_logits = 3  # 3: c, w, obj
        
        self.emb_V = Identity()
        self.emb_A = Identity()
        self.pos_enc_V = PositionalEncoder(cfg.d_model_video, cfg.dout_p)
        self.pos_enc_A = PositionalEncoder(cfg.d_model_audio, cfg.dout_p)

        # load the pre-trained encoder from captioning module
        self.encoder = BiModalEncoder(
            cap_cfg.d_model_audio, cap_cfg.d_model_video, cap_cfg.d_model, 
            cap_cfg.dout_p, cap_cfg.H, cap_cfg.d_ff_audio, 
            cap_cfg.d_ff_video, cap_cfg.N
        )
        
        self.encoder.load_state_dict(encoder_weights)
        self.encoder = self.encoder.to(cfg.device)
        
        dims_A = [cfg.d_model_audio, *cfg.conv_layers_audio, self.num_logits*cfg.anchors_num_audio]
        dims_V = [cfg.d_model_video, *cfg.conv_layers_video, self.num_logits*cfg.anchors_num_video]
        self.detection_layers_A = torch.nn.ModuleList([
            ProposalGenerationHead(dims_A, k, cfg.dout_p) for k in cfg.kernel_sizes['audio']
        ])
        self.detection_layers_V = torch.nn.ModuleList([
            ProposalGenerationHead(dims_V, k, cfg.dout_p) for k in cfg.kernel_sizes['video']
        ])

    def forward_modality(self, x, detection, stride, anchors_list):
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
        anchors_tensor = torch.tensor(anchors_list, device=self.cfg.device)
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
                Av, layer, self.cfg.strides['audio'], self.anchors['audio']
            )
            total_loss_A += loss_A
            all_predictions_A.append(props_A)
            sum_losses_dict_A = add_dict_to_another_dict(losses_A, sum_losses_dict_A)

        for layer in self.detection_layers_V:
            props_V, loss_V, losses_V = self.forward_modality(
                Va, layer, self.cfg.strides['video'], self.anchors['video']
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

def add_dict_to_another_dict(one_dict, another_dict):
    another_dict = {k: another_dict.get(k, 0) + v for k, v in one_dict.items()}
    return another_dict