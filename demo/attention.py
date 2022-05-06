import torch
import streamlit as st

from thumt.utils import HParams
from thumt.data import M30kDataset
from thumt.models import get_model
from thumt.bin.translator import default_params, merge_params, import_params, override_params

root = '/data/private/mxy/mmt'
data_path = f'{root}/data/multi30k-dataset/data/task1'
vocab_path = f'{root}/data/wmt18/de-en'
exp_path = f'{root}/exp/THUMT'

def default_args():
    test_name = 'test_2016_flickr'
    return HParams(
        models = ['visual_prefix_transformer_v6'],
        input = [f'{data_path}/raw/{test_name}.32k.en'],
        img_input = [f'{data_path}/image_splits', f'{data_path}/vit-all.pkl'],
        output = None,
        vocabulary = [f'{vocab_path}/vocab.32k.en.txt', f'{vocab_path}/vocab.32k.de.txt'],
        checkpoints = [f'{exp_path}/pretrain/wmt18+cc-en-de-big/eval'],
        prefix = f'{exp_path}/visual-prefix-tuning_v6/wmt18+cc-en-de-big/eval/model-1.pt',
        parameters = 'device_list=[0],max_length=64',
        hparam_set = 'big'
    )

def load_model(params, model_cls):
    print("Loading Transformer model...", flush=True)
    transformer_model_cls = get_model('transformer')
    transformer_model = transformer_model_cls(params).cuda()
    print("Finished.", flush=True)

    model = model_cls(transformer_model, params).cuda()

    return model

def load_dataset(params, args):
    split = args.input[0].split('/')[-1].split('.')[0]
    dataset = M30kDataset(args.input[0], params.img_input, params.vocabulary, params.device, split=split)
    return dataset

def main(args):
    model_cls_list = [get_model(model) for model in args.models]
    params_list = [default_params() for _ in range(len(model_cls_list))]
    params_list = [
        merge_params(params, model_cls.default_params(args.hparam_set))
        for params, model_cls in zip(params_list, model_cls_list)]
    params_list = [
        import_params(args.checkpoints[i], args.models[i], params_list[i])
        for i in range(len(args.checkpoints))]
    for i in range(len(args.models)):
        if 'prefix_transformer' in args.models[i]:
            params_list[i] = import_params(f"{args.prefix.split('/model-')[0]}", args.models[i], params_list[i])
    params_list = [
        override_params(params_list[i], args)
        for i in range(len(model_cls_list))]

    params = params_list[0]
    model_cls = model_cls_list[0]

    dataset = load_dataset(params, args)

    with torch.no_grad():
        model = load_model(params, model_cls)

        feature = dataset[0]

        import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    args = default_args()

    main(args)