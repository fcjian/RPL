import json
import os
import argparse
from PIL import Image
from tqdm import tqdm
from concurrent import futures


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate the object crops from LVIS')
    parser.add_argument('--file-path', help='path of the annotation file')
    parser.add_argument('--img-root', help='path of the image')
    parser.add_argument('--save-root', help='path to save the object crops')
    parser.add_argument('--scaling-factor', type=float, default=1.0,
                        help='factor for extending the object box')
    parser.add_argument('--min-size', type=float, default=20.0,
                        help='the min size of the box')
    parser.add_argument('--num-thread', type=int, default=10,
                        help='the number of the thread to save the images')
    args = parser.parse_args()
    return args


def cropping_fun(image_anns, img_root, save_root, catid2names, imageid2names, scaling_factor, min_size):
    for i in tqdm(range(len(image_anns)), total=len(image_anns)):
        image_ann = image_anns[i]
        cat_id = image_ann['category_id']
        img_id = image_ann['image_id']

        cat_name = catid2names[cat_id]
        img_name = imageid2names[img_id]
        img_path = os.path.join(img_root, img_name)
        save_cat_root = os.path.join(save_root, cat_name)
        save_path = os.path.join(save_cat_root, str(i) + '_' + img_name.split('/')[-1])

        x, y, w, h = image_ann['bbox']
        if w * h < min_size:
            print(f"skip {save_path.split('/')[-1]} with size {w * h}!")
            continue

        image = Image.open(img_path)
        xmin = max(0, int(x - scaling_factor * w))
        ymin = max(0, int(y - scaling_factor * h))
        xmax = min(image.size[0], int(x + (1 + scaling_factor) * w))
        ymax = min(image.size[1], int(y + (1 + scaling_factor) * h))
        image = image.crop([xmin, ymin, xmax, ymax])

        os.makedirs(save_cat_root, exist_ok=True)
        try:
            image.save(save_path)
        except:
            print(f"{save_path.split('/')[-1]} can not be saved!")

    return True


def main():
    args = parse_args()

    file_path = args.file_path
    img_root = args.img_root
    save_root = args.save_root
    scaling_factor = args.scaling_factor
    min_size = args.min_size
    num_thread = args.num_thread

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    with open(file_path, 'r') as f:
        load_dict = json.load(f)

    image_infos = load_dict['images']
    imageid2names = {}
    for i in range(len(image_infos)):
        image_info = image_infos[i]
        phase, img_name = image_info['coco_url'].split('/')[-2:]
        imageid2names[image_info['id']] = os.path.join(phase, img_name)

    cat_infos = load_dict['categories']
    catid2names = {}
    for i in range(len(cat_infos)):
        cat_info = cat_infos[i]
        catid2names[cat_info['id']] = cat_info['name']

    image_anns = load_dict['annotations']

    count_per_thread = (len(image_anns) + num_thread - 1) // num_thread
    image_anns = [image_anns[i * count_per_thread:(i + 1) * count_per_thread] for i in range(num_thread)]

    with futures.ThreadPoolExecutor(max_workers=num_thread) as executor:
        threads = [executor.submit(cropping_fun, image_ann, img_root=img_root, save_root=save_root, \
                                   catid2names=catid2names, imageid2names=imageid2names, \
                                   scaling_factor=scaling_factor, min_size=min_size) for image_ann in image_anns]
        for future in futures.as_completed(threads):
            print(future.result())


if __name__ == '__main__':
    main()
