import torch
import timm
import skimage.io as io
import clip
import argparse
import pickle
import os

from PIL import Image
from tqdm import tqdm
from glob import glob
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

dataroot = '/data/private/mxy/mmt/data/multi30k-dataset/data/task1'

data2path = {
    'flickr': f'{dataroot}/flickr30k-images',
    'test_2017_flickr': f'{dataroot}/test2017',
    'test_2018_flickr': f'{dataroot}/test2018',
    'test_2017_mscoco': '/data/private/mxy/mmt/code/fairseq_mmt/flickr30k/testcoco-images'
}


def load_model(model_type, device='cpu'):
    print(f"Loading vit {model_type} model...")
    if 'clip' in model_type:
        model_type = model_type.replace('clip-', '')
        model, preprocess = clip.load(model_type, device=device, jit=False)
    elif 'timm' in model_type:
        model_type = model_type.replace('timm-', '')
        model = timm.create_model(model_type, pretrained=True, num_classes=0).to(device)
        config = resolve_data_config({}, model=model)
        preprocess = create_transform(**config)
    else:
        print(f'No such model: {model_type}')
    model.eval()
    print("Done")
    return model, preprocess

def main(model_type, dataset):
    device = torch.device('cuda:0')
    model_name = model_type.replace('/', '_')

    image_path = data2path[dataset]
    filenames = glob(f"{image_path}/*.jpg")

    model, preprocess = load_model(model_type, device)

    out_path = f"/data/private/mxy/mmt/data/multi30k-dataset/data/task1/{dataset}-{model_name}"

    all_features = {}
    for img_path in tqdm(filenames):
        img_name = img_path.split('/')[-1].split('#')[0]
        img_id = img_name[:-4]

        # image = io.imread(img_path)
        # image = Image.fromarray(image)
        img = Image.open(img_path).convert("RGB")
        img = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            img_feature = model.encode_image(img).cpu() if 'clip' in model_type else model.forward_features(img)

        all_features[img_id] = img_feature.cpu()

    with open(f"{out_path}.pkl", 'wb') as f:
        pickle.dump(all_features, f)

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="clip-ViT-B/32", choices=('clip-ViT-B/32', 'timm-vit_base_patch16_384'))
    parser.add_argument('--dataset', default="flickr", choices=('flickr', 'test_2017_flickr', 'test_2018_flickr', 'test_2017_mscoco'))
    args = parser.parse_args()

    main(args.model_type, args.dataset)
