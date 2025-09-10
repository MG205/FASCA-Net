import torch
from tools.function import dict_to_str, device, EnTrainer
from tools.dataloarder import MultiModalDataset
from models.model import FASCA_Net
import random
from torch.utils.data import DataLoader
import numpy as np


def EnRun(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    train_ds = MultiModalDataset(split='train')
    valid_ds = MultiModalDataset(split='valid')
    test_ds = MultiModalDataset(split='test')

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)

    model = FASCA_Net(config).to(device)

    trainer = EnTrainer(config)

    for epoch in range(config.epoch_num):
        print(f'---------------------EPOCH: {epoch+1}--------------------')
        trainer.do_train(model, train_loader)
        trainer.do_test(model, val_loader, "VAL")
        # trainer.do_test(model, test_loader, "TEST")

    print("\n------------🚀TEST🚀----------------\n")
    save_path = r"checkpoint/best_model.pt"
    model = torch.load(save_path, map_location=device)
    trainer.do_test(model, test_loader, "TEST")
