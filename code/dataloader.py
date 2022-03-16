import torch
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor
from transformers import AutoTokenizer

import os
from PIL import Image
import json


class CC(Dataset):
    def __init__(self, info_file, image_dir, split, num_character=2):
        self.info_file = info_file
        self.image_dir = image_dir
        self.split = split
        self.num_character = num_character

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.data = []
        self.load_data()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def load_data(self):
        transform = PILToTensor()
        info_file = open(self.info_file)
        data_info = json.load(info_file)
        for sample in data_info[self.split]:
            # Load Images
            image_path = os.path.join(self.image_dir, self.split, sample['id'] + '.png')
            image = transform(Image.open(image_path)) / 255.0

            # Tokenize Labels
            label = sample['label']
            label = self.tokenizer(text=label,
                                   padding='max_length',
                                   max_length=self.num_character,
                                   add_special_tokens=False,
                                   return_attention_mask=False)
            label = torch.tensor(label['input_ids'])
            self.data.append({'image': image,  # (3, H, W)
                              'label': label   # (num_character, )
                              })


if __name__ == '__main__':
    train_set = CC('../data_image/data_info.json', '../data_image/captcha_20/', 'test', 2)
    pass
