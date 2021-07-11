import torch.nn as nn
 
from video.model.blocks import (Identity,
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

        self.emb_C.init_word_embeddings(train_dataset.train_vocab.vectors, cfg.unfreeze_word_emb)

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