import os
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm


class MultiModalDataset(Dataset):
    def __init__(self, split='train', transform=None):
        """
        split (str): 'train', 'valid' or 'test'
        transform (callable, optional):
        """
        base_dir = os.path.dirname(os.path.abspath(os.getcwd()))
        pkl_path = os.path.join(base_dir, "mml/data", "aligned_50.pkl")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        assert split in data, f"Split {split} not exit， plase try again：{list(data.keys())}"
        split_data = data[split]

        self.audio = split_data['audio']
        self.vision = split_data['vision']
        self.text = split_data['text']
        self.labels = split_data['classification_labels']
        self.transform = transform

        n = len(self.labels)
        assert len(self.audio) == n and len(self.vision) == n and len(self.text) == n, "audio, vision, test error!"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_feat = torch.tensor(self.audio[idx], dtype=torch.float32)
        vision_feat = torch.tensor(self.vision[idx], dtype=torch.float32)
        text_feat = torch.tensor(self.text[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        audio_mask = (audio_feat.abs().sum(dim=-1) == 0).int()
        vision_mask = (vision_feat.abs().sum(dim=-1) == 0).int()
        text_mask = (text_feat.abs().sum(dim=-1) == 0).int()

        sample = {
            'audio':       audio_feat.float(),
            'vision':      vision_feat.float(),
            'text':        text_feat.float(),
            'audio_mask':  audio_mask.float(),
            'vision_mask': vision_mask.float(),
            'text_mask':   text_mask.float(),
            'label':       label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
