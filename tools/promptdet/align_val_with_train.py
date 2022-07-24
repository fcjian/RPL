import os
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Align the categories of the val set with the one of the training set')
    parser.add_argument('--train-root', help='path of the training data')
    parser.add_argument('--val-root', help='path of the val data')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    train_root = args.train_root
    val_root = args.val_root

    class_names = os.listdir(train_root)
    for name in tqdm(class_names, total=len(class_names)):
        os.makedirs(os.path.join(val_root, name), exist_ok=True)


if __name__ == '__main__':
    main()
