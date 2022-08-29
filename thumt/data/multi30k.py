import torch
import pickle
import skimage.io as io

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from thumt.data.pipeline import _sort_input_file
from thumt.tokenizers import WhiteSpaceTokenizer

def get_infer_dataset(filename, params, model_name, preprocess, dtype, raw=False):
    sorted_key, sorted_data = _sort_input_file(filename)
    split = filename.split('/')[-1].split('.')[0]

    sorted_keys = {v:k for k,v in sorted_key.items()}

    if model_name == 'visual_prefix_transformer_v2' or model_name == 'visual_prefix_transformer_v4':
        dataset = M30kDatasetv2(sorted_data, params.img_input, 
                                params.vocabulary, params.device, preprocess,
                                dtype, params.max_length, params.bos, params.eos, 
                                params.pad, params.unk, split, sorted_keys)
    else:
        dataset = M30kDataset(sorted_data, params.img_input, 
                              params.vocabulary, params.device,
                              params.max_length, params.bos, params.eos, 
                              params.pad, params.unk, split, sorted_keys, raw)
        

    return sorted_key, dataset

class M30kDataset(Dataset):
    def __init__(self,
                 txt_input,
                 img_input,
                 vocab,
                 device,
                 seq_len=64,
                 bos=b'<bos>',
                 eos=b'<eos>',
                 pad=b'<pad>',
                 unk=b'<unk>',
                 split='train',
                 sorted_keys=None,
                 raw=False,
                 fewshot_name=None
        ):
        self.bos = bos
        self.eos = eos
        self.pad = pad
        self.unk = unk
        self.seq_len = seq_len
        self.split = split
        self.device = device

        self.tokenizer = WhiteSpaceTokenizer()
        self.src_vocab = vocab['source']
        self.tgt_vocab = vocab['target']

        self.sorted_keys = sorted_keys
        self.raw = raw
        self.fewshot_name = fewshot_name

        if fewshot_name is not None:
            print(f'Using few-shot setting: {fewshot_name}')

        self.pad_id = self.src_vocab[pad]
        self.unk_id = self.src_vocab[unk]
        
        if sorted_keys is not None:
            self.src_txt, self.src_raw = self.load_text(txt_input, self.src_vocab, None, self.eos)
        else:
            self.src_txt, self.src_raw = self.load_text(txt_input[0], self.src_vocab, None, self.eos)

        if split == 'train':
            self.tgt_txt, self.tgt_raw = self.load_text(txt_input[1], self.tgt_vocab, self.bos, None)
            self.lbl_txt, self.lbl_raw = self.load_text(txt_input[1], self.tgt_vocab, None, self.eos)

        self.img_ids, self.img_features = self.load_image_features(img_input[0], img_input[1])

        assert len(self.src_txt) == len(self.img_ids)

    def __len__(self):
        return len(self.src_txt)

    def __getitem__(self, idx):
        src_seq = torch.tensor(self.src_txt[idx])
        src_mask = (src_seq != self.pad_id).float()

        src_raw = self.src_raw[idx]

        # if the dataset is sorted
        if self.sorted_keys is not None:
            idx = self.sorted_keys[idx]

        img_id = self.img_ids[idx]
        img_feature = self.img_features[img_id]

        features = {
            "img_feature": img_feature.cuda(self.device).float(),
            "source": src_seq.cuda(self.device),
            "source_mask": src_mask.cuda(self.device)
        }

        if self.raw:
            features.update({
                "raw_source": src_raw,
                "imgid": img_id,
            })

        if self.split == 'train':
            tgt_seq = torch.tensor(self.tgt_txt[idx])
            lbl_seq = torch.tensor(self.lbl_txt[idx])
            tgt_mask = (tgt_seq != self.pad_id).float()
            features.update({
                "target": tgt_seq.cuda(self.device),
                "target_mask": tgt_mask.cuda(self.device)
            })
            return features, lbl_seq.cuda(self.device)

        return features

    def load_text(self, txt_input, vocab, bos=None, eos=None):
        sentences = []
        raw = []
        if isinstance(txt_input, str):
            with open(txt_input, 'rb') as fin:
                lines = fin.read().splitlines()
        elif isinstance(txt_input, list):
            lines = txt_input
        else:
            import ipdb; ipdb.set_trace()
            raise LookupError(f"Unknown txt input type {type(txt_input)}")

        for line in lines:
            sent = self.tokenizer.encode(line)

            if bos:
                sent.insert(0, bos)

            if eos:
                sent.append(eos)
            
            tokens = [self.pad_id] * self.seq_len
            for i, s in enumerate(sent):
                if s in vocab:
                    tokens[i] = vocab[s]
                else:
                    tokens[i] = self.unk_id

                if i == self.seq_len - 1:
                    if eos:
                        tokens[i] = vocab[eos]
                    break
            
            sentences.append(tokens)
            raw.append(line)
        return sentences, raw

    def load_image_features(self, filepath, feature_path):
        if self.split == 'train':
            if self.fewshot_name is None:
                fn = f'{filepath}/{self.split}.txt.shuf'
            else:
                fn = f'{filepath}/{self.fewshot_name}.txt'
        else:
            fn = f'{filepath}/{self.split}.txt'
            # if self.fewshot_name is not None and 'dog' in self.fewshot_name:
            #     fn = f'{filepath}/{self.fewshot_name}.txt'

        with open(fn, 'r') as fin:
            img_names = fin.read().splitlines()
            if '#' in img_names[0]:
                img_names = [n.split('#')[0] for n in img_names]
        img_ids = [name[:-4] for name in img_names]

        with open(feature_path, 'rb') as fin:
            all_features = pickle.load(fin)

        img_features = {k:all_features[k] for k in img_ids}

        return img_ids, img_features

class M30kDatasetv2(Dataset):
    def __init__(self, 
                 txt_input, 
                 img_input, 
                 vocab, 
                 device,
                 preprocess,
                 dtype,
                 seq_len=64, 
                 bos=b'<bos>', 
                 eos=b'<eos>', 
                 pad=b'<pad>', 
                 unk=b'<unk>',
                 split='train',
                 sorted_keys=None
        ):
        self.bos = bos
        self.eos = eos
        self.pad = pad
        self.unk = unk
        self.seq_len = seq_len
        self.split = split
        self.device = device
        self.dtype = dtype

        self.tokenizer = WhiteSpaceTokenizer()
        self.src_vocab = vocab['source']
        self.tgt_vocab = vocab['target']

        self.sorted_keys = sorted_keys

        self.pad_id = self.src_vocab[pad]
        self.unk_id = self.src_vocab[unk]
        
        if sorted_keys is not None:
            self.src_txt = self.load_text(txt_input, self.src_vocab, None, self.eos)
        else:
            self.src_txt = self.load_text(txt_input[0], self.src_vocab, None, self.eos)

        if split == 'train':
            self.tgt_txt = self.load_text(txt_input[1], self.tgt_vocab, self.bos, None)
            self.lbl_txt = self.load_text(txt_input[1], self.tgt_vocab, None, self.eos)

        self.img_ids, self.images = self.load_images(img_input[0], img_input[1], preprocess)

        assert len(self.src_txt) == len(self.img_ids)

    def __len__(self):
        return len(self.src_txt)

    def __getitem__(self, idx):
        src_seq = torch.tensor(self.src_txt[idx])
        src_mask = (src_seq != self.pad_id).float()

        # if the dataset is sorted
        if self.sorted_keys is not None:
            idx = self.sorted_keys[idx]

        img_id = self.img_ids[idx]
        image = self.images[img_id]

        features = {
            "image": image.cuda(self.device),
            "source": src_seq.cuda(self.device),
            "source_mask": src_mask.cuda(self.device)
        }

        if self.split == 'train':
            tgt_seq = torch.tensor(self.tgt_txt[idx])
            lbl_seq = torch.tensor(self.lbl_txt[idx])
            tgt_mask = (tgt_seq != self.pad_id).float()
            features.update({
                "target": tgt_seq.cuda(self.device),
                "target_mask": tgt_mask.cuda(self.device)
            })
            return features, lbl_seq.cuda(self.device)

        return features

    def load_text(self, txt_input, vocab, bos=None, eos=None):
        sentences = []
        if isinstance(txt_input, str):
            with open(txt_input, 'rb') as fin:
                lines = fin.read().splitlines()
        elif isinstance(txt_input, list):
            lines = txt_input
        else:
            import ipdb; ipdb.set_trace()
            raise LookupError(f"Unknown txt input type {type(txt_input)}")

        for line in lines:
            sent = self.tokenizer.encode(line)

            if bos:
                sent.insert(0, bos)

            if eos:
                sent.append(eos)
            
            tokens = [self.pad_id] * self.seq_len
            for i, s in enumerate(sent):
                if s in vocab:
                    tokens[i] = vocab[s]
                else:
                    tokens[i] = self.unk_id

                if i == self.seq_len - 1:
                    if eos:
                        tokens[i] = vocab[eos]
                    break
            
            sentences.append(tokens)
        return sentences

    def load_images(self, filepath, imgpath, preprocess):
        if self.split == 'train':
            fn = f'{filepath}/{self.split}.txt.shuf'
        else:
            fn = f'{filepath}/{self.split}.txt'
        with open(fn, 'r') as fin:
            img_names = fin.read().splitlines()
            if '#' in img_names[0]:
                img_names = [n.split('#')[0] for n in img_names]
        img_ids = [name[:-4] for name in img_names]

        images = {}
        for imgid in tqdm(img_ids, desc='Loading images'):
            if 'coco' in imgpath:
                img_name = f'{imgpath}/train2014/{imgid}.jpg' if 'train' in imgid else f'{imgpath}/val2014/{imgid}.jpg'
            else:
                img_name = f'{imgpath}/{imgid}.jpg'
            image = io.imread(img_name)
            images[imgid] = preprocess(Image.fromarray(image)).type(self.dtype)

        return img_ids, images