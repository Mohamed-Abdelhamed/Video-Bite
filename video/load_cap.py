from torch.utils.data.dataset import Dataset
import torch
import pickle
from video.model.captioning_module import BiModalTransformer


class LoadCapModel():
    def load(self, pretrained_cap_model_path, device) -> tuple:
        cap_model_cpt = torch.load(pretrained_cap_model_path, map_location='cpu')
        cfg = cap_model_cpt['config']
        cfg.device = device
        cfg.pretrained_cap_model_path = pretrained_cap_model_path

        # load train dataset just for special token's indices
        train_dataset = ActivityNetCaptionsDataset(cfg)

        # define model and load the weights
        model = BiModalTransformer(cfg, train_dataset)
        model = torch.nn.DataParallel(model, [device])
        model.load_state_dict(cap_model_cpt['model_state_dict'])  # if IncompatibleKeys - ignore
        model.eval()

        return cfg, model, train_dataset 


class ActivityNetCaptionsDataset(Dataset):
    
    def __init__(self, cfg):
        self.train_vocab = caption_iterator()
        
        self.trg_voc_size = len(self.train_vocab)
        self.pad_idx = self.train_vocab.stoi[cfg.pad_token]
        self.start_idx = self.train_vocab.stoi[cfg.start_token]
        self.end_idx = self.train_vocab.stoi[cfg.end_token]



def caption_iterator():
    print(f'Contructing caption_iterator for train phase')
    
    with open("video/sample/vocab.pth", 'rb') as file:
        while True:
            try:
                train_vocab = pickle.load(file)
            except EOFError:
                break

    return train_vocab
