from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import cv2
import pytesseract
import numpy as np
import torch
import time
import random

to_tensor = transforms.ToTensor()


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bert_tokens, hate = sample['image'], sample["bert_tokens"], sample['class']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        """image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        image = image.float()
        image /= 255"""
        image = to_tensor(image)
        bert_tokens = torch.LongTensor(bert_tokens)  # torch.tensor(bert_tokens, dtype=torch.long)
        hate = torch.FloatTensor(hate)  # torch.tensor(hate)

        sample["image"] = image
        sample["bert_tokens"] = bert_tokens
        sample["class"] = hate

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        # landmarks = landmarks - [left, top]
        sample["image"] = image

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample["image"]
        image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_CUBIC)
        sample["image"] = image

        return sample


class Tokenize(object):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sample):
        tokens = self.tokenizer.tokenize(sample["text"])

        tokens = tokens[:48]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        sample["bert_tokens"] = input_ids

        return sample


class HateWordsVector(object):

    def __init__(self, list):
        self.hateWords = list
        self.total_hate_hate = 0
        self.total_hate_love = 0
        self.total_sentences_hate = 0
        self.empty_text = 0

    def __call__(self, sample):

        text = sample["text"]
        if text.strip() == '':
            self.empty_text += 1

        N = len(self.hateWords)

        vector = torch.zeros(N)

        hate_found = False

        for i in range(N):
            if self.hateWords[i] in text:
                vector[i] = 1
                self.total_hate_hate += sample["class"][0]  # TODO converting labels to list
                self.total_hate_hate += 1 - sample["class"][0]  # TODO converting labels to list

                hate_found = True

        if hate_found:
            self.total_sentences_hate += 1

        sample["hate_words"] = vector

        return sample


class ImagesDataLoader(Dataset):

    def __init__(self, metadata, base_path, transform=None, max_amount=1000):
        self.transform = transform
        self.base_path = base_path
        print(base_path + '/' + metadata)
        self.data = pd.read_json(base_path + '/' + metadata, lines=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.base_path + "/" + self.data.at[index, 'img']
        text = self.data.at[index, 'text']
        label = self.data.at[index, 'label']

        image = cv2.imread(img_path)

        assert image is not None

        text = text.replace("\n", " ")
        sample = {'image': image, "text": text, "class": [label], "image_path": img_path}  # TODO [label] ???

        if self.transform:
            sample = self.transform(sample)

        return sample


class UnsupervisedMatcherDataLoader(Dataset):
    def __init__(self, metadata, base_path, transform=None, max_amount=1000):

        self.transform = transform
        self.base_path = base_path

        paths = [f"{base_path}/{path}" for path in meta]

        total = []

        for x in paths:
            x = x.strip("\n")
            total.append([x, x + '.ocr', 0.0])

            for i in range(1):
                y = random.choice(paths)
                y = y.strip("\n")
                total.append([x, y + '.ocr', 1.0])

        self.data = total

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):
        path = self.base_path + "/" + self.data[index][0]
        text_path = self.base_path + "/" + self.data[index][1]
        image = cv2.imread(path)

        assert image is not None

        try:
            text = open(text_path)
            text = text.read()
        except:
            text = pytesseract.image_to_string(image, config='--oem 1')
            print("WARNING: PREVIOUS OCR EXTRACTION NOT FOUND. THIS SLOWS DOWN THE DATA LOADING by 2000%")
            print(text_path)
        text = text.replace("\n", " ")

        hate = self.data[index][2]

        sample = {'image': image, "text": text, "class": hate, "image_path": path}

        if self.transform:
            sample = self.transform(sample)

        return sample


def custom_collate(batch):
    batch2 = {
        "image": batch[0]["image"].unsqueeze(0),
        "class": batch[0]["class"].unsqueeze(0),
        "hate_words": batch[0]["hate_words"].unsqueeze(0),
    }

    image_paths = [x["image_path"] for x in batch]

    batch2["image_paths"] = image_paths

    tokens = [batch[0]["bert_tokens"]]

    for b in batch[1:]:
        batch2["image"] = torch.cat((batch2["image"], b["image"].unsqueeze(0)))
        batch2["class"] = torch.cat((batch2["class"], b["class"].unsqueeze(0)))
        batch2["hate_words"] = torch.cat((batch2["hate_words"], b["hate_words"].unsqueeze(0)))
        tokens.append(b["bert_tokens"])

    try:
        batch2["bert_tokens"] = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)

    except Exception as e:
        print(e)
        print(tokens)
        quit()

    del batch

    return batch2
