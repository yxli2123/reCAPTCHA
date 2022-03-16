import os
import random
import json


if __name__ == '__main__':
    random.seed(0)
    fp = open('data_info.json', 'r')
    fp1 = open('data_info.json', 'w')
    data_info = json.load(fp)
    data_len = len(data_info)

    new_data_info = {'train': [],
                     'valid': [],
                     'test': []}

    random.shuffle(data_info)

    os.mkdir('captcha_20/train')
    os.mkdir('captcha_20/valid')
    os.mkdir('captcha_20/test')

    for sample in data_info[0: int(0.7 * data_len)]:
        new_data_info['train'].append(sample)
        os.system(f'mv ./captcha_20/{sample["id"]}.png ./captcha_20/train')
    for sample in data_info[int(0.7 * data_len): int(0.8 * data_len)]:
        new_data_info['valid'].append(sample)
        os.system(f'mv ./captcha_20/{sample["id"]}.png ./captcha_20/valid')
    for sample in data_info[int(0.8 * data_len):]:
        new_data_info['test'].append(sample)
        os.system(f'mv ./captcha_20/{sample["id"]}.png ./captcha_20/test')

    json.dump(new_data_info, fp1, indent=4)
