
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import pytesseract
import numpy as np
import torch
import time
import random

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image, bert_tokens, hate = sample['image'], sample["bert_tokens"], sample['class']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        image = image.float()
        image /= 255

        bert_tokens = torch.tensor(bert_tokens, dtype=torch.long)
        hate = torch.tensor(hate)

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

        tokens = tokens[:50]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)



        # self.dataloader.max_length = max(self.dataloader.max_length, len(input_ids))

        sample["bert_tokens"] = input_ids

        return sample

class HateWordsVector(object):

    def __init__(self, list):
        self.hateWords = list


    def __call__(self, sample):

        text = sample["text"]

        N = len(self.hateWords)

        vector = torch.zeros(N)

        for i in range(N):
            if self.hateWords[i] in text:
                vector[i] = 1

        sample["hate_words"] = vector

        return sample




class ImagesDataLoader(Dataset):
    def __init__(self, love_metadata, hate_metadata, base_path, transform=None):


        ## Keys:
        # image
        # text
        # bert_tokens
        # hate_words

        # class

        self.transform = transform
        self.base_path = base_path

        love_paths = list(open(base_path + "/" + love_metadata))
        hate_paths = list(open(base_path + "/" + hate_metadata))

        love_paths = [[x.strip("\n"), 0.0] for x in love_paths]
        hate_paths = [[x.strip("\n"), 1.0] for x in hate_paths]

        self.data = love_paths + hate_paths

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        # t = time.time()

        path = self.base_path + "/" + self.data[index][0]
        text_path = path + ".ocr"
        # print(path)
        image = cv2.imread(path)
        # print("imread:", time.time() - t)
        # t = time.time()

        assert image is not None

        try:
            text = open(text_path)
            text = text.read()
        except:
            text = pytesseract.image_to_string(image, config='--oem 1')
            # print("pytesseract time, ", time.time() - t)
            print("WARNING: PREVIOUS OCR EXTRACTION NOT FOUND. THIS SLOWS DOWN THE DATA LOADING by 2000%")
        text = text.replace("\n", " ")
        # print(text)

        hate = self.data[index][1]

        sample = {'image': image, "text": text, "class": hate}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ImageTextMatcherDataLoader(Dataset):
    def __init__(self, love_metadata, hate_metadata, base_path, transform=None, max_amount = 1000):


        ## Keys:
        # image
        # text
        # bert_tokens
        # hate_words

        # class

        self.transform = transform
        self.base_path = base_path

        love_paths = list(open(base_path + "/" + love_metadata))
        hate_paths = list(open(base_path + "/" + hate_metadata))

        total = []

        for x in hate_paths:
            # print(x)
            x = x.strip("\n")

            y = random.choice(love_paths)
            # print(y)
            y = y.strip("\n")

            total.append([x, x+'.ocr', 0.0])
            total.append([x, y+'.ocr', 1.0])

        for x in love_paths:
            x = x.strip("\n")

            y = random.choice(hate_paths)
            y = y.strip("\n")

            total.append([x, x + '.ocr', 0.0])
            total.append([x, y + '.ocr', 1.0])

        self.data = total




    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        # t = time.time()

        path = self.base_path + "/" + self.data[index][0]
        text_path = self.base_path + "/" + self.data[index][1]
        # print(path)
        image = cv2.imread(path)
        # print("imread:", time.time() - t)
        # t = time.time()

        assert image is not None

        try:
            text = open(text_path)
            text = text.read()
        except:
            text = pytesseract.image_to_string(image, config='--oem 1')
            # print("pytesseract time, ", time.time() - t)
            print("WARNING: PREVIOUS OCR EXTRACTION NOT FOUND. THIS SLOWS DOWN THE DATA LOADING by 2000%")
            print(text_path)
        text = text.replace("\n", " ")
        # print(text)

        hate = self.data[index][2]

        sample = {'image': image, "text": text, "class": hate}

        if self.transform:
            sample = self.transform(sample)

        return sample

def custom_collate(batch):

    batch2 = {"image": batch[0]["image"].unsqueeze(0),
              "class": batch[0]["class"].unsqueeze(0),
              "hate_words": batch[0]["hate_words"].unsqueeze(0)}

    tokens = [batch[0]["bert_tokens"]]

    for b in batch[1:]:
        batch2["image"] = torch.cat((batch2["image"], b["image"].unsqueeze(0)))
        batch2["class"] = torch.cat((batch2["class"], b["class"].unsqueeze(0)))
        batch2["hate_words"] = torch.cat((batch2["hate_words"], b["hate_words"].unsqueeze(0)))
        tokens.append(b["bert_tokens"])

    batch2["bert_tokens"] = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)


    del batch

    return batch2



