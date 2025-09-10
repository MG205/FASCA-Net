import argparse
from tools.function import EnConfig
from tools.train import EnRun
import warnings
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False.*",
    category=UserWarning
)


def main(args):
    assert args.dataset == 'mosi', "dataset error, try again!"
    EnRun(EnConfig(batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed, model=args.model
                   , dataset_name=args.dataset, num_hidden_layers=args.num_hidden_layers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--epoch_num', type=int, default=3, help='epoch of train')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--model', type=str, default='cme', help='concatenate(cc) or cross-modality encoder')
    parser.add_argument('--dataset', type=str, default='mosi', help='dataset name: mosi, mosei, sims')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='number of hidden layers for cross-modality encoder')
    args = parser.parse_args()
    main(args)





