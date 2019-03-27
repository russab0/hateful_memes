
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import pytesseract

import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image, text, hate = sample['image'], sample["text"], sample['class']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        image = image.float()
        image /= 255

        text = torch.tensor(text, dtype=torch.long)

        print(text.size())
        print(text)
        quit()
        text_full = torch.zeros()

        return {'image': image,
                'class': torch.tensor(hate),
                'text': text}


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

    def __init__(self, tokenizer, dataloader):

        self.tokenizer = tokenizer
        # self.dataloader = dataloader

    def __call__(self, sample):

        tokens = self.tokenizer.tokenize(sample["text"])
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # self.dataloader.max_length = max(self.dataloader.max_length, len(input_ids))

        sample["text"] = input_ids

        return sample

class ImagesDataLoader(Dataset):
    def __init__(self, love_metadata, hate_metadata, base_path, transform=None):

        self.transform = transform
        self.base_path = base_path

        love_paths = list(open(base_path + "/" + love_metadata))
        hate_paths = list(open(base_path + "/" + hate_metadata))

        love_paths = [[x.strip("\n"), 0.0] for x in love_paths]
        hate_paths = [[x.strip("\n"), 1.0] for x in hate_paths]

        self.data = love_paths + hate_paths

        self.max_length = -1

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        path = self.base_path + "/" + self.data[index][0]
        # print(path)
        image = cv2.imread(path)
        text = pytesseract.image_to_string(image)
        text = text.replace("\n", "")

        assert image is not None

        hate = self.data[index][1]

        sample = {'image': image, "text": text, "class": hate}

        if self.transform:
            sample = self.transform(sample)

        return sample


# if __name__ == '__main__':
#
#     transform = transforms.Compose([Rescale((224, 224)),
#                                     ToTensor()])
#
#     images_dataset = ImagesDataLoader("imagesList.txt", "imagesList.txt", "data/hate_memes_GI", transform)
#
#     dataloader = DataLoader(images_dataset, batch_size=10, shuffle=True)
#
#     for batch in dataloader:
#
#         print(type(batch))
#
#         input_batch = batch["image"]
#         target_batch = batch["class"]



