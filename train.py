import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from module import test

import time


class MultimodalClassifier(nn.Module):
    def __init__(self, im_feat_model):
        super(MultimodalClassifier, self).__init__()

        self.im_feat_model = im_feat_model

    def forward(self, x):


        image_features = self.im_feat_model(x)

        return image_features


if __name__ == '__main__':

    start_time = time.time()

    # Configuring CUDA / CPU execution
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    # Get image descriptor
    VGG16_features = torchvision.models.vgg16(pretrained=True)
    VGG16_features.to(device)

    full_model = MultimodalClassifier(VGG16_features)
    full_model.to(device)


    transform = transforms.Compose([test.Rescale((224, 224)),
                                    test.ToTensor()])

    images_dataset = test.ImagesDataLoader("redditMemesList.txt", "hateMemesList.txt", "data/train_data", transform)

    dataloader = DataLoader(images_dataset, batch_size=10, shuffle=True)

    for batch in dataloader:

        # print("batch")
        input_batch = batch["image"].to(device)
        target_batch = batch["class"].to(device)

        pred = full_model(input_batch)

        # print(pred)
        # print(pred.shape)

    end_time = time.time()
    print("Elapsed Time:", end_time - start_time)

