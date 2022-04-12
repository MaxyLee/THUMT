import torch
import skimage.io as io
import clip
import argparse
import pickle
import os

from PIL import Image
from tqdm import tqdm


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    testname= 'mscoco'
    # img_path = f"/data1/private/mxy/projects/mmt/data/multi30k-dataset/data/task1/{testname}"
    # image_path = "/data2/share/data/coco2014"
    image_path = "/data2/share/data/flickr30k-entities/flickr30k-images"
    out_path = f"/data1/private/mxy/projects/mmt/data/multi30k-dataset/data/task1/{testname}-{clip_model_name}"
    os.makedirs(out_path, exist_ok=True)

    print(f"Loading clip {clip_model_type} model...")
    # clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    print("Done")
    import ipdb; ipdb.set_trace()
    filenames = glob(f"{image_path}/*.jpg")
    # img_split = '/data1/private/mxy/projects/mmt/data/multi30k-dataset/data/task1/image_splits/test_2017_mscoco.txt'
    # with open(img_split, 'r') as f:
    #     filenames = [s.split('#')[0] for s in f.read().splitlines()]

    all_features = {}
    for img_name in tqdm(filenames):
        img_name = img_path.split('/')[-1]
        img_id = img_name[:-4]
        # img_path = f'{image_path}/train2014/{img_name}' if 'train' in img_name else f'{image_path}/val2014/{img_name}'

        image = io.imread(img_path)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)

        with torch.no_grad():
            img_feature = clip_model.encode_image(image).cpu()

        all_features[img_id] = img_feature

    with open(f"{out_path}.pkl", 'wb') as f:
        pickle.dump(all_features, f)

    print('Done')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
