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
        print(f'Pretrained caption path: \n {cfg.pretrained_cap_model_path}')

        # load train dataset just for special token's indices
        train_dataset = ActivityNetCaptionsDataset(cfg)

        # define model and load the weights
        model = BiModalTransformer(cfg, train_dataset)
        model = torch.nn.DataParallel(model, [device])
        model.load_state_dict(cap_model_cpt['model_state_dict'])  # if IncompatibleKeys - ignore
        model.eval()

        encoder_weights = {k: v for k, v in cap_model_cpt['model_state_dict'].items() if 'encoder' in k}
        encoder_weights = {k.replace('module.encoder.', ''): v for k, v in encoder_weights.items()}

        return cfg, model, train_dataset, encoder_weights


class ActivityNetCaptionsDataset(Dataset):
    
    def __init__(self, cfg):
        with open("video/sample/vocab.pth", 'rb') as file:
            while True:
                try:
                    self.train_vocab = pickle.load(file)
                except EOFError:
                    break
        
        self.trg_voc_size = len(self.train_vocab)
        self.pad_idx = self.train_vocab.stoi[cfg.pad_token]
        self.start_idx = self.train_vocab.stoi[cfg.start_token]
        self.end_idx = self.train_vocab.stoi[cfg.end_token]