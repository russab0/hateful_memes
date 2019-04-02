import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


from module import test

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

import time

from tensorboardX import SummaryWriter


class MultimodalClassifier(nn.Module):
    def __init__(self, image_feat_model, text_feat_model, TOTAL_FEATURES, hidden_size):

        super(MultimodalClassifier, self).__init__()
        self.im_feat_model = image_feat_model
        self.text_feat_model = text_feat_model

        self.classifier = nn.Sequential(
            nn.Linear(TOTAL_FEATURES, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            # nn.Softmax()
        )




    def forward(self, image, text):


        image_features = self.im_feat_model(image)
        text_features = self.text_feat_model(text)

        text_features = text_features[1]

        # cat_size = image_features.size()[1] + text_features.size()[1]

        fusion = torch.cat((image_features, text_features), dim=1)

        out = self.classifier(fusion)

        return out


if __name__ == '__main__':

    start_time = time.time()
    writer = SummaryWriter("logs/testing3")

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
    bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
    bert_model.to(device)


    # VGG16 output --> 1000 features
    # BERT text embedder --> 768 features
    # Total = 1768

    IMAGE_AND_TEXT_FEATURES = 1768
    HIDDEN_SIZE = 1000
    N_EPOCHS = 3

    full_model = MultimodalClassifier(VGG16_features, bert_model,
                                      IMAGE_AND_TEXT_FEATURES, HIDDEN_SIZE)
    full_model.to(device)

    transform = transforms.Compose([test.Rescale((224, 224)),
                                    test.Tokenize(tokenizer),
                                    test.ToTensor()])

    images_dataset = test.ImagesDataLoader("redditMemesList.txt", "hateMemesList.txt", "data/train_data", transform)

    dataloader = DataLoader(images_dataset, batch_size=10, shuffle=True, collate_fn=test.custom_collate)


    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(full_model.classifier.parameters(), lr=0.01, momentum=0.9)

    iteration = 0

    for i in range(N_EPOCHS):

        for batch in dataloader:



            print("batch")
            image_batch = batch["image"].to(device)
            text_batch = batch["text"].to(device)
            target_batch = batch["class"].to(device)
            target_batch = target_batch.unsqueeze(1)


            optimizer.zero_grad()

            pred = full_model(image_batch, text_batch)

            #
            # print((pred.type()))
            # print((target_batch.type()))
            # quit()

            # pred = pred.long()
            # target_batch = target_batch.long()

            loss = criterion(pred, target_batch)

            loss.backward()

            optimizer.step()

            writer.add_scalar('logs/scalar1', loss, iteration)
            iteration += 1

            print(pred)
            print(loss)

    end_time = time.time()
    print("Elapsed Time:", end_time - start_time)

    writer.close()

