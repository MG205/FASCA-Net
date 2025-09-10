import os
import torch
from torch import nn
from tqdm import tqdm
from tools.metricsTop import MetricsTop

# device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f" % (key, src_dict[key])
    return dst_str


class EnConfig(object):
    def __init__(
        self,
        epoch_num=10,
        train_mode: str = 'regression',
        model_save_path: str = 'checkpoint/',
        learning_rate: float = 1e-5,
        dataset_name: str = 'mosi',
        early_stop: int = 8,
        seed: int = 0,
        dropout: float = 0.3,
        model: str = 'cc',
        batch_size: int = 16,
        model_size: str = 'small',
        num_hidden_layers: int = 1,
    ):
        self.train_mode = train_mode
        self.learning_rate = learning_rate
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.early_stop = early_stop
        self.seed = seed
        self.dropout = dropout
        self.model = model
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.model_size = model_size
        self.num_hidden_layers = num_hidden_layers


class EnTrainer:
    def __init__(self, config: EnConfig):
        self.config = config
        # choose loss based on mode
        if config.train_mode == 'regression':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        # metrics object
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)

        # record best Has0_acc_2
        self.best_has0_acc2 = float("-inf")
        # ensure save directory exists
        os.makedirs(self.config.model_save_path, exist_ok=True)

    def do_train(self, model: nn.Module, data_loader):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        total_loss = 0.0

        for batch in tqdm(data_loader, desc="TRAIN"):
            text_inputs = batch['text'].to(device)
            text_mask = batch['text_mask'].to(device)
            audio_inputs = batch['audio'].to(device)
            audio_mask = batch['audio_mask'].to(device)
            vision_inputs = batch['vision'].to(device)
            vision_mask = batch['vision_mask'].to(device)

            targets = batch['label'].to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(
                text_inputs, text_mask,
                audio_inputs, audio_mask,
                vision_inputs, vision_mask
            )
            # assume 'M' is the multimodal output key
            loss = self.criterion(outputs['M'], targets)
            total_loss += loss.item() * text_inputs.size(0)

            loss.backward()
            optimizer.step()

        avg_loss = round(total_loss / len(data_loader.dataset), 4)
        print(f"TRAIN >> loss: {avg_loss:.4f}")
        return avg_loss

    def do_test(self, model: nn.Module, data_loader, mode: str):
        model.eval()
        y_pred, y_true = [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=mode.upper()):
                text_inputs = batch['text'].to(device)
                text_mask = batch['text_mask'].to(device)
                audio_inputs = batch['audio'].to(device)
                audio_mask = batch['audio_mask'].to(device)
                vision_inputs = batch['vision'].to(device)
                vision_mask = batch['vision_mask'].to(device)

                targets = batch['label'].to(device).view(-1, 1)
                outputs = model(text_inputs, text_mask, audio_inputs, audio_mask, vision_inputs, vision_mask)
                loss = self.criterion(outputs['M'], targets)
                total_loss += loss.item() * text_inputs.size(0)

                y_pred.append(outputs['M'].cpu())
                y_true.append(targets.cpu())

        avg_loss = round(total_loss / len(data_loader.dataset), 4)
        print(f"{mode} >> loss: {avg_loss:.4f}")

        pred = torch.cat(y_pred)
        true = torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        print(f"M >>{dict_to_str(eval_results)}")
        eval_results['Loss'] = avg_loss

        curr = eval_results.get('Has0_acc_2', None)
        if curr is None:
            print("\nWarning: 'Has0_acc_2' not found in eval_results; skipping save.")
        else:
            if curr > self.best_has0_acc2:
                self.best_has0_acc2 = curr
                save_path = os.path.join(self.config.model_save_path, 'best_model.pt')
                torch.save(model, save_path)
                print(f"\nNew best Has0_acc_2: {curr:.4f}. Model saved to {save_path}.")
            else:
                print(f"\nHas0_acc_2: {curr:.4f} (best: {self.best_has0_acc2:.4f}). Skip saving.")

        return eval_results
