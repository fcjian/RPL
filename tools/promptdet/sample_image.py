import os
import argparse
import random
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sample the images using symbolic links')
    parser.add_argument('--source-root', help='path of the source images')
    parser.add_argument('--target-root', help='target path to link the source image')
    parser.add_argument('--num-images', type=int, default=200,
                       help='the number of the training images per catogory')
    parser.add_argument('--random-sample', action='store_true',
                        help='whether to random sample')
    parser.add_argument('--laion-image', action='store_true',
                        help='whether to link to the laion images')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    source_root = args.source_root  # "temp/lvis_object_crops"
    target_root = args.target_root # "temp/lvis_and_laion_data"
    num_images = args.num_images # 200
    random_sample = args.random_sample # True
    laion_image = args.laion_image # False

    dir_names = os.listdir(source_root)
    dir_names = [dir_name for dir_name in dir_names if '.txt' not in dir_name]
    for dir_name in tqdm(dir_names, total=len(dir_names)):
        if laion_image:
            source_path = os.path.join(source_root, dir_name, "00000")
        else:
            source_path = os.path.join(source_root, dir_name)
        target_path = os.path.join(target_root, dir_name)

        file_names = os.listdir(source_path)
        file_names = [file_name for file_name in file_names if 'jpg' in file_name]
        if random_sample:
            random.shuffle(file_names)
        else:
            file_names.sort()

        for file_name in file_names[:num_images]:
            os.makedirs(target_path, exist_ok=True)
            os.symlink(os.path.abspath(os.path.join(source_path, file_name)), os.path.join(target_path, file_name))


if __name__ == '__main__':
    main()
