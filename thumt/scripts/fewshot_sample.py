import random

dataroot = '/data1/private/mxy/projects/mmt/data/multi30k-dataset/data/task1'
seed = 18

sample_nums = [50, 100, 200, 500]
sample_times = 5

if __name__ == '__main__':
    random.seed(seed)

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
