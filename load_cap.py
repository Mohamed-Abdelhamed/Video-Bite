from model.blocks import FeatureEmbedder, Identity, PositionalEncoder, VocabularyEmbedder #THIS
from model.encoders import BiModalEncoder #THIS
from model.decoders import BiModelDecoder #THIS

import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torch
import spacy
from torchtext import data
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

class LoadCapModel():
    def load(self, pretrained_cap_model_path, device) -> tuple:
        cap_model_cpt = torch.load(pretrained_cap_model_path, map_location='cpu')
        cfg = cap_model_cpt['config']
        cfg.device = device
        cfg.pretrained_cap_model_path = pretrained_cap_model_path
        cfg.train_meta_path = './video/data/train.csv'

        # load train dataset just for special token's indices
        train_dataset = ActivityNetCaptionsDataset(cfg, 'train', get_full_feat=False)

        # define model and load the weights
        model = BiModalTransformer(cfg, train_dataset)
        model = torch.nn.DataParallel(model, [device])
        model.load_state_dict(cap_model_cpt['model_state_dict'])  # if IncompatibleKeys - ignore
        model.eval()

        return cfg, model, train_dataset 


class ActivityNetCaptionsDataset(Dataset):
    
    def __init__(self, cfg, phase, get_full_feat):

        '''
            For the doc see the __getitem__.
        '''
        self.cfg = cfg
        self.phase = phase
        self.get_full_feat = get_full_feat

        self.feature_names = f'{cfg.video_feature_name}_{cfg.audio_feature_name}'
        
        self.meta_path = cfg.train_meta_path
        self.batch_size = cfg.train_batch_size
        
        # caption dataset *iterator*
        self.train_vocab, self.caption_loader = caption_iterator(cfg, self.batch_size, self.phase)
        
        self.trg_voc_size = len(self.train_vocab)
        self.pad_idx = self.train_vocab.stoi[cfg.pad_token]
        self.start_idx = self.train_vocab.stoi[cfg.start_token]
        self.end_idx = self.train_vocab.stoi[cfg.end_token]
    
        # initialize the caption loader iterator
        self.caption_loader_iter = iter(self.caption_loader)


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


def caption_iterator(cfg, batch_size, phase):
    print(f'Contructing caption_iterator for "{phase}" phase')
    spacy_en = spacy.load('en')
    
    CAPTION = data.ReversibleField(
        tokenize='spacy', init_token=cfg.start_token, eos_token=cfg.end_token, 
        pad_token=cfg.pad_token, lower=True, batch_first=True, is_target=True
    )
    INDEX = data.Field(
        sequential=False, use_vocab=False, batch_first=True
    )
    
    # the order has to be the same as in the table
    fields = [
        ('video_id', None),
        ('caption', CAPTION),
        ('start', None),
        ('end', None),
        ('duration', None),
        ('phase', None),
        ('idx', INDEX),
    ]

    dataset = data.TabularDataset(
        path=cfg.train_meta_path, format='tsv', skip_header=True, fields=fields,
    )
    CAPTION.build_vocab(dataset.caption, min_freq=cfg.min_freq_caps, vectors=cfg.word_emb_caps)
    train_vocab = CAPTION.vocab

    # sort_key = lambda x: data.interleave_keys(len(x.caption), len(y.caption))
    datasetloader = data.BucketIterator(dataset, batch_size, sort_key=lambda x: 0, 
                                        device=torch.device(cfg.device), repeat=False, shuffle=True)
    return train_vocab, datasetloader
