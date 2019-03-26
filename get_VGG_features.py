import torchvision
import torch.nn as nn
import torch

import cv2
import torchvision.transforms as transforms

import time
import requests
import random


BATCH_SHAPE = (244, 244)

class DataLoader:
    def __init__(self, metadata_path, device, shuffle=False):

        self.subpath = '/'.join(metadata_path.split('/')[:-1])
        # self.subpath = '.'

        self.metadata = [x.strip("\n") for x in open(metadata_path)]
        # print(self.metadata)

        self.device = device
        if shuffle:
            random.shuffle(self.metadata)

    def get_len(self):
        return len(self.metadata)


    def get_batch(self, size):

        src_batch = -1

        paths = []


        for i, x in enumerate(self.metadata):

            if i == 0:

                path = self.subpath + '/' + x

                # print(path)


                src_batch = cv2.imread(path)

                src_batch = cv2.resize(src_batch, BATCH_SHAPE, interpolation=cv2.INTER_CUBIC)

                src_batch = torch.from_numpy(src_batch).permute(2, 0, 1).unsqueeze(0).float()

                src_batch = src_batch/255

                src_batch = src_batch.to(self.device)
                self.metadata = self.metadata[1:]

            else:

                path = self.subpath + '/' + x
                srcim = cv2.imread(path)

                srcim = cv2.resize(srcim, BATCH_SHAPE, interpolation=cv2.INTER_CUBIC)
                srcim = torch.from_numpy(srcim).permute(2, 0, 1).unsqueeze(0).float()
                srcim = srcim / 255
                srcim = srcim.to(self.device)
                src_batch = torch.cat((src_batch, srcim))
                self.metadata = self.metadata[1:]

            paths.append(x)

            if i == size - 1:
                break

        return paths, src_batch



if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    VGG16_features = torchvision.models.vgg16(pretrained=True)

    VGG16_features.to(device)

    # VGG16_features.classifier = VGG16_features.classifier[:2]

    BATCH_SIZE = 10

    metadata_path = "data/downloads/imagesList.txt"
    # metadata_path = "images_list.txt"
    output_folder_path = "data/downloads/VGGfeatures"
    # output_folder_path = "VGGfeatures"

    dataloader = DataLoader(metadata_path, device)

    paths, src_batch = dataloader.get_batch(BATCH_SIZE)


    while type(src_batch) != type(-1):

        with torch.no_grad():
            prediction = VGG16_features(src_batch)

        print("batch")

        for i in range(prediction.size()[0]):
            features = prediction[i]

            path = paths[i]

            path = path.split("/")[-1]

            torch.save(features, output_folder_path + '/' + path + ".vggf")

        paths, src_batch = dataloader.get_batch(BATCH_SIZE)

    # im = cv2.imread("data/reddit/memes/2bvcu7.jpg")
    # im = torch.from_numpy(im)
    #
    # im = im.permute(2, 0, 1)
    #
    #
    #
    # im = im.float()
    # im = im/256
    #
    # print("kk3", im.mean())
    # time.sleep(1)
    #
    #
    #
    #
    # im = im.unsqueeze(0)
    # print(im.shape)
    #
    #
    # output = VGG16_features(im)
    #
    # print(output.size(), output)
    #
    # LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
    #
    # # Let's get our class labels.
    # response = requests.get(LABELS_URL)  # Make an HTTP GET request and store the response.
    # labels = {int(key): value for key, value in response.json().items()}
    #
    # prediction = output.data.numpy().argmax()
    #
    # print(prediction)
    # print(labels[prediction])