import streamlit as st

from PIL import Image
from glob import glob

data_path = '/data1/private/mxy/projects/mmt/data/multi30k-dataset/data/task1'

def load_image(image_path, imgid):
    if 'coco' in image_path:
        imgid = imgid.split('#')[0]
        if 'train' in imgid:
            img_fn = f'{image_path}/train2014/{imgid}'
        else:
            img_fn = f'{image_path}/val2014/{imgid}'
    else:
        img_fn = f'{image_path}/{imgid}'
    img = Image.open(img_fn).convert('RGB')
    return img

############ ui funcs ############

@st.cache
def load_dataset(split):
    if split == 'test_2016_flickr':
        image_path = f'{data_path}/flickr30k-images'
    elif split == 'test_2017_flickr':
        image_path = f'{data_path}/test2017'
    elif split == 'test_2018_flickr':
        image_path = f'{data_path}/test2018'
    elif split == 'test_2017_mscoco':
        image_path = '/data2/share/data/coco2014'
    else:
        print(f'no such split: {split}')

    with open(f'{data_path}/image_splits/{split}.txt') as f:
        image_ids = f.read().splitlines()
    
    with open(f'{data_path}/raw/{split}.en') as f:
        en = f.read().splitlines()

    with open(f'{data_path}/raw/{split}.de') as f:
        de = f.read().splitlines()

    trans_filenames = glob(f'{data_path}/raw/{split}-*.trans.norm')

    trans = {}
    for fn in trans_filenames:
        with open(fn) as f:
            trans[fn] = f.read().splitlines()

    return image_path, image_ids, en, de, trans
    
def select_index(length):
    index = st.sidebar.selectbox(label='Selected Index', options=range(length))
    return index
    
def select_dataset(splits):
    split = st.sidebar.selectbox(label='Selected Split', options=splits)
    return split

if __name__ == '__main__':
    splits = ['test_2016_flickr', 'test_2017_flickr', 'test_2018_flickr', 'test_2017_mscoco']

    split = select_dataset(splits)
    image_path, image_ids, en, de, trans = load_dataset(split)

    # display
    idx = select_index(len(image_ids))
    imgid = image_ids[idx]

    st.write(f'image name: {imgid}')

    img = load_image(image_path, imgid)
    st.image(img, clamp=True, output_format="PNG")

    st.write(f'En')
    st.markdown(f'`{en[idx]}`')
    st.write(f'De')
    st.markdown(f'`{de[idx]}`')

    for fn, tran in trans.items():
        name = fn.split('/')[-1].split('-')[-1].strip('.trans.norm')

        st.write()
        st.write(f'Translation from {name}')
        st.write()
        st.markdown(f'`{tran[idx]}`')