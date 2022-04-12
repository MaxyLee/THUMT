import torch
import pickle
import skimage.io as io

from torch import nn
from PIL import Image
from tqdm import tqdm
from clip.model import LayerNorm, Transformer

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        return x

def extract_all_features():
    embed_dim = 512
    vision_width = 768
    vision_layers = 12
    vision_patch_size = 32
    grid_size = 7
    vision_heads = vision_width // 64
    image_resolution = vision_patch_size * grid_size

    vit = VisionTransformer(image_resolution, vision_patch_size, vision_width, vision_layers, vision_heads, embed_dim)

    ckpt_path = '/data1/private/mxy/projects/mmt/exp/clip/vit.pt'
    state_dict = torch.load(ckpt_path)

    vit.load_state_dict(state_dict)

    preprocess = _transform(image_resolution)

    dataset_names = [
                     'train', 
                     'val', 
                     'test_2016_flickr', 
                     'test_2017_flickr', 
                     'test_2018_flickr', 
                     'test_2017_mscoco'
                    ]
    image_paths = [
                   "/data2/share/data/flickr30k-entities/flickr30k-images", 
                   "/data2/share/data/flickr30k-entities/flickr30k-images",
                   "/data2/share/data/flickr30k-entities/flickr30k-images",
                   "/data1/private/mxy/projects/mmt/data/multi30k-dataset/data/task1/test2017",
                   "/data1/private/mxy/projects/mmt/data/multi30k-dataset/data/task1/test2018",
                   "/data2/share/data/coco2014"
                   ]
    fn_path = '/data1/private/mxy/projects/mmt/data/multi30k-dataset/data/task1/image_splits'
    for dn, ip in zip(dataset_names, image_paths):
        out_path = f"/data1/private/mxy/projects/mmt/data/multi30k-dataset/data/task1/vit-{dn}.pkl"
        with open(f'{fn_path}/{dn}.txt') as f:
            filenames = f.read().splitlines()

        all_features = {}
        for img_name in tqdm(filenames, desc=dn):
            if 'COCO' in img_name:
                img_name = img_name.split('#')[0]
                if 'train' in img_name:
                    img_path = f'{ip}/train2014/{img_name}'
                else:
                    img_path = f'{ip}/val2014/{img_name}'
            else:
                img_path = f'{ip}/{img_name}'

            img_id = img_name[:-4]
            
            image = io.imread(img_path)
            image = preprocess(Image.fromarray(image)).unsqueeze(0)

            with torch.no_grad():
                img_feature = vit(image)
            
            all_features[img_id] = img_feature.squeeze()
        
        with open(out_path, 'wb') as f:
            pickle.dump(all_features, f)

if __name__ == '__main__':
    extract_all_features()