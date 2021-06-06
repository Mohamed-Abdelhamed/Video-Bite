import torch
import torch.nn as nn

from video.model.encoders import Encoder

from video.model.blocks import FeatureEmbedder, Identity, PositionalEncoder


class ProposalGenerationHead(nn.Module):

    def __init__(self, d_model_list, kernel_size, dout_p):
        super(ProposalGenerationHead, self).__init__()
        assert kernel_size % 2 == 1, 'It is more convenient to use odd kernel_sizes for padding'
        conv_layers = []
        in_dims = d_model_list[:-1]
        out_dims = d_model_list[1:]
        N_layers = len(d_model_list) - 1

        for n, (in_d, out_d) in enumerate(zip(in_dims, out_dims)):
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
        

class ProposalGenerator(nn.Module):
    
    def __init__(self, cfg, anchors):
        super(ProposalGenerator, self).__init__()
        self.cfg = cfg
        self.EPS = 1e-16
        self.num_logits = 3  # 3: c, w, obj
        self.anchors = anchors
        self.anchors_list = anchors[cfg.modality]
        self.anchors_num = len(self.anchors_list)

        if cfg.modality == 'video':
            self.d_feat = cfg.d_vid
            self.d_model_modality = cfg.d_model_video
            self.d_ff = cfg.d_ff_video
            layer_dims = [
                self.d_model_modality, *cfg.conv_layers_video, self.num_logits*self.anchors_num
            ]
        elif cfg.modality == 'audio':
            self.d_feat = cfg.d_aud
            self.d_model_modality = cfg.d_model_audio
            self.d_ff = cfg.d_ff_audio
            layer_dims = [
                self.d_model_modality, *cfg.conv_layers_audio, self.num_logits*self.anchors_num
            ]
        else:
            raise NotImplementedError

        if cfg.use_linear_embedder:
            self.emb = FeatureEmbedder(self.d_feat, self.d_model_modality)
        else:
            self.emb = Identity()
        self.pos_enc = PositionalEncoder(self.d_model_modality, cfg.dout_p)

        self.encoder = Encoder(self.d_model_modality, cfg.dout_p, cfg.H, self.d_ff, cfg.N)
        # encoder initialization
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.detection_layers = torch.nn.ModuleList([
            ProposalGenerationHead(layer_dims, k, cfg.dout_p) for k in cfg.kernel_sizes[cfg.modality]
        ])

        print(self.detection_layers)
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def kernel_size_forward(self, x, layer, stride, targets):
        # in case targets is None
        loss = 0
        losses = {}
        x = layer(x)

        B, S, D = x.shape
        x = x.view(B, S, self.anchors_num, self.num_logits)

        x = x.permute(0, 2, 1, 3).contiguous()
        grid_cell = torch.arange(S).view(1, 1, S).float().to(self.cfg.device)
        # After dividing anchors by the stride, they represent the size size of
        # how many grid celts they are overlapping: 1.2 = 1 and 20% of a grid cell.
        # After multiplying them by the stride, the pixel values are going to be
        # obtained.
        anchors_list = [[anchor / stride] for anchor in self.anchors_list]
        anchors_tensor = torch.tensor(anchors_list, device=self.cfg.device)
        # (A, 2) -> (1, A, 1) for broadcasting
        prior_length = anchors_tensor.view(1, self.anchors_num, 1)

        # prediction values for the *loss* calculation (training)
        sigma_c = torch.sigmoid(x[:, :, :, 0])  # center shift
        l = x[:, :, :, 1]  # log coefficient
        sigma_o = torch.sigmoid(x[:, :, :, 2])  # objectness

        # prediction values that are going to be used for the original image
        # we need to detach them from the graph as we don't need to backproparate on them
        predictions = x.clone().detach()
        # broadcasting (B, A, S) + (1, 1, S)
        predictions[:, :, :, 0] = sigma_c + grid_cell
        # broadcasting (1, A, 1) * (B, A, S)
        predictions[:, :, :, 1] = prior_length * torch.exp(l)
        predictions[:, :, :, 2] = sigma_o

        # for NMS: (B, A, S, 3) -> (B, A*S, 3)
        predictions = predictions.view(B, S*self.anchors_num, self.num_logits)
        predictions[:, :, :2] *= stride
    
        return predictions, loss, losses
        
    def forward(self, x, targets, masks):

        if self.cfg.modality == 'video':
            x = x['rgb'] + x['flow']
            stride = self.cfg.strides['video']
            x = self.emb(x)
            x = self.pos_enc(x)
            x = self.encoder(x, masks['V_mask'])
        elif self.cfg.modality == 'audio':
            x = x['audio']
            stride = self.cfg.strides['audio']
            x = self.emb(x)
            x = self.pos_enc(x)
            x = self.encoder(x, masks['A_mask'])

        all_predictions = []
        # total_loss should have backward
        sum_losses_dict = {}
        total_loss = 0
        
        for layer in self.detection_layers:
            predictions, loss, loss_dict = self.kernel_size_forward(x, layer, stride, targets)
            total_loss += loss
            all_predictions.append(predictions)
            sum_losses_dict = add_dict_to_another_dict(loss_dict, sum_losses_dict)

        all_predictions = torch.cat(all_predictions, dim=1)

        return all_predictions, total_loss, sum_losses_dict
    

def add_dict_to_another_dict(one_dict, another_dict):
    another_dict = {k: another_dict.get(k, 0) + v for k, v in one_dict.items()}
    return another_dict