import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from module import test

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

import time


class MultimodalClassifier(nn.Module):
    def __init__(self, image_feat_model):

        super(MultimodalClassifier, self).__init__()
        self.im_feat_model = image_feat_model

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


    # To embed text, we use a Pytorch implementation of BERT: Using pythorch BERT implementation from https://github.com/huggingface/pytorch-pretrained-BERT
    # Get Textual Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case="true")
    # Get Textual Embedding.
    model = BertModel.from_pretrained("bert-base-multilingual-cased")



    full_model = MultimodalClassifier(VGG16_features)
    full_model.to(device)

    dataloader = None
    transform = transforms.Compose([test.Rescale((224, 224)),
                                    test.Tokenize(tokenizer, dataloader),
                                    test.ToTensor()])

    images_dataset = test.ImagesDataLoader("redditMemesList.txt", "hateMemesList.txt", "data/train_data", transform)

    dataloader = DataLoader(images_dataset, batch_size=10, shuffle=True)

    for batch in dataloader:

        # print("batch")
        input_batch = batch["image"].to(device)
        text_batch = batch["text"].to(device)
        target_batch = batch["class"].to(device)

        print(text_batch)

        quit()

        pred = full_model(input_batch)

        # print(pred)
        # print(pred.shape)

    end_time = time.time()
    print("Elapsed Time:", end_time - start_time)

