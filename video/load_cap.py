from torch.utils.data.dataset import Dataset
import torch
import spacy
from torchtext import data
from video.model.captioning_module import BiModalTransformer


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
