import random

dataroot = '/data1/private/mxy/projects/mmt/data/multi30k-dataset/data/task1'
seed = 18

sample_nums = [50, 100, 200, 500]
sample_times = 5

def normal_sample():
    with open(f'{dataroot}/image_splits/train.txt', 'r') as f:
        img_ids = f.read().splitlines()

    with open(f'{dataroot}/raw/train.32k.en', 'r') as f:
        en = f.read().splitlines()

    with open(f'{dataroot}/raw/train.32k.de', 'r') as f:
        de = f.read().splitlines()

    for num in sample_nums:
        for i in range(sample_times):
            name = f'train-fs{num}-{i}'

            sample_ids = random.sample(range(len(img_ids)), num)

            if len(set(sample_ids)) != num:
                print(f'Duplicate found!')
                exit()

            fs_img_ids = [img_ids[id] for id in sample_ids]
            fs_en = [en[id] for id in sample_ids]
            fs_de = [de[id] for id in sample_ids]

            with open(f'{dataroot}/image_splits/{name}.txt', 'w') as fout:
                for img_id in fs_img_ids:
                    fout.write(img_id + '\n')

            with open(f'{dataroot}/raw/{name}.32k.en', 'w') as fout:
                for line in fs_en:
                    fout.write(line + '\n')

            with open(f'{dataroot}/raw/{name}.32k.de', 'w') as fout:
                for line in fs_de:
                    fout.write(line + '\n')

def keyword_sample(keyword, split=False):
    with open(f'{dataroot}/image_splits/train.txt', 'r') as f:
        img_ids = f.read().splitlines()

    with open(f'{dataroot}/raw/train.32k.en', 'r') as f:
        en = f.read().splitlines()

    with open(f'{dataroot}/raw/train.32k.de', 'r') as f:
        de = f.read().splitlines()

    sample_ids = []

    for i, sent in enumerate(en):
        tokens = sent.split(' ')
        if keyword in tokens:
            sample_ids.append(i)

    kw_img_ids = [img_ids[id] for id in sample_ids]
    kw_en = [en[id] for id in sample_ids]
    kw_de = [de[id] for id in sample_ids]

    with open(f'{dataroot}/image_splits/val.{keyword}.txt', 'w') as fout:
        for img_id in kw_img_ids[-100:-50]:
            fout.write(img_id + '\n')

    with open(f'{dataroot}/raw/val.{keyword}.32k.en', 'w') as fout:
        for line in kw_en[-100:-50]:
            fout.write(line + '\n')

    with open(f'{dataroot}/raw/val.{keyword}.32k.de', 'w') as fout:
        for line in kw_de[-100:-50]:
            fout.write(line + '\n')

    with open(f'{dataroot}/image_splits/test.{keyword}.txt', 'w') as fout:
        for img_id in kw_img_ids[-50:]:
            fout.write(img_id + '\n')

    with open(f'{dataroot}/raw/test.{keyword}.32k.en', 'w') as fout:
        for line in kw_en[-50:]:
            fout.write(line + '\n')

    with open(f'{dataroot}/raw/test.{keyword}.32k.de', 'w') as fout:
        for line in kw_de[-50:]:
            fout.write(line + '\n')

    if split:
        num = 50
        for i in range(sample_times):
            name = f'train-{keyword}{num}-{i}'

            sample_ids = random.sample(range(len(kw_img_ids) - 100), num)

            if len(set(sample_ids)) != num:
                print(f'Duplicate found!')
                exit()

            fs_img_ids = [kw_img_ids[id] for id in sample_ids]
            fs_en = [kw_en[id] for id in sample_ids]
            fs_de = [kw_de[id] for id in sample_ids]

            with open(f'{dataroot}/image_splits/{name}.txt', 'w') as fout:
                for img_id in fs_img_ids:
                    fout.write(img_id + '\n')

            with open(f'{dataroot}/raw/{name}.32k.en', 'w') as fout:
                for line in fs_en:
                    fout.write(line + '\n')

            with open(f'{dataroot}/raw/{name}.32k.de', 'w') as fout:
                for line in fs_de:
                    fout.write(line + '\n')
    

if __name__ == '__main__':
    random.seed(seed)

    # normal_sample()

    keyword = 'dog'
    keyword_sample(keyword, split=True)