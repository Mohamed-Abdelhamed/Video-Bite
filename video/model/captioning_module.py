import torch
import torch.nn as nn
 
from video.model.blocks import (FeatureEmbedder, Identity,
                          PositionalEncoder, VocabularyEmbedder)
from video.model.decoders import BiModelDecoder
from video.model.encoders import BiModalEncoder
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, d_model, voc_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, voc_size)
        print('Using vanilla Generator')

    def forward(self, x):
        '''
        Inputs:
            x: (B, Sc, Dc)
        Outputs:
            (B, seq_len, voc_size)
        '''
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)


class BiModalTransformer(nn.Module):
    '''
    Forward:
        Inputs:
            src {'rgb'&'flow' (B, Sv, Dv), 'audio': (B, Sa, Da)}
            trg (C): ((B, Sc))
            masks: {'V_mask': (B, 1, Sv), 'A_mask': (B, 1, Sa), 'C_mask' (B, Sc, Sc))}
        Output:
            C: (B, Sc, Vc)
    '''
    def __init__(self, cfg, train_dataset):
        super(BiModalTransformer, self).__init__()

        if cfg.use_linear_embedder:
            self.emb_A = FeatureEmbedder(cfg.d_aud, cfg.d_model_audio)
            self.emb_V = FeatureEmbedder(cfg.d_vid, cfg.d_model_video)
        else:
            self.emb_A = Identity()
            self.emb_V = Identity()

        self.emb_C = VocabularyEmbedder(train_dataset.trg_voc_size, cfg.d_model_caps)
        
        self.pos_enc_A = PositionalEncoder(cfg.d_model_audio, cfg.dout_p)
        self.pos_enc_V = PositionalEncoder(cfg.d_model_video, cfg.dout_p)
        self.pos_enc_C = PositionalEncoder(cfg.d_model_caps, cfg.dout_p)

        self.encoder = BiModalEncoder(
            cfg.d_model_audio, cfg.d_model_video, cfg.d_model, cfg.dout_p, cfg.H, 
            cfg.d_ff_audio, cfg.d_ff_video, cfg.N
        )
        
        self.decoder = BiModelDecoder(
            cfg.d_model_audio, cfg.d_model_video, cfg.d_model_caps, cfg.d_model, cfg.dout_p, 
            cfg.H, cfg.d_ff_caps, cfg.N
        )

        self.generator = Generator(cfg.d_model_caps, train_dataset.trg_voc_size)

        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # initialize embedding after, so it will replace the weights
        # of the prev. initialization
        self.emb_C.init_word_embeddings(train_dataset.train_vocab.vectors, cfg.unfreeze_word_emb)

        # load the pretrained encoder from the proposal (used in ablation studies)
        if cfg.pretrained_prop_model_path is not None:
            print(f'Pretrained prop path: \n {cfg.pretrained_prop_model_path}')
            cap_model_cpt = torch.load(cfg.pretrained_prop_model_path, map_location='cpu')
            encoder_config = cap_model_cpt['config']
            self.encoder = BiModalEncoder(
                encoder_config.d_model_audio, encoder_config.d_model_video, encoder_config.d_model, 
                encoder_config.dout_p, encoder_config.H, encoder_config.d_ff_audio, 
                encoder_config.d_ff_video, encoder_config.N
            )
            encoder_weights = {k: v for k, v in cap_model_cpt['model_state_dict'].items() if 'encoder' in k}
            encoder_weights = {k.replace('encoder.', ''): v for k, v in encoder_weights.items()}
            self.encoder.load_state_dict(encoder_weights)
            self.encoder = self.encoder.to(cfg.device)
            for param in self.encoder.parameters():
                param.requires_grad = cfg.finetune_prop_encoder

    def forward(self, src: dict, trg, masks: dict):
        V, A = src['rgb'] + src['flow'], src['audio']
        C = trg

        # (B, Sm, Dm) <- (B, Sm, Dm), m in [a, v]; 
        A = self.emb_A(A)
        V = self.emb_V(V)
        # (B, Sc, Dc) <- (S, Sc)
        C = self.emb_C(C)
        
        A = self.pos_enc_A(A)
        V = self.pos_enc_V(V)
        C = self.pos_enc_C(C)
        
        # notation: M1m2m2 (B, Sm1, Dm1), M1 is the target modality, m2 is the source modality
        Av, Va = self.encoder((A, V), masks)

        # (B, Sc, Dc)
        C = self.decoder((C, (Av, Va)), masks)
        
        # (B, Sc, Vc) <- (B, Sc, Dc) 
        C = self.generator(C)

        return C
