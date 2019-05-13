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

from tqdm import tqdm


class MultimodalClassifier(nn.Module):
    def __init__(self, image_feat_model, text_feat_model, TOTAL_FEATURES,
                 USE_IMAGE, USE_TEXT, hidden_size):

        super(MultimodalClassifier, self).__init__()
        self.im_feat_model = image_feat_model
        self.text_feat_model = text_feat_model

        self.USE_IMAGE = USE_IMAGE
        self.USE_TEXT = USE_TEXT

        self.classifier = nn.Sequential(
            nn.Linear(TOTAL_FEATURES, hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, 1),
            # nn.Softmax()
        )




    def forward(self, image, text):
        with torch.no_grad():
            if self.USE_IMAGE == 1:
                image_features = self.im_feat_model(image)
                features = image_features

            if self.USE_TEXT == 1:
                text_features = self.text_feat_model(text)
                text_features = text_features[1]
                features = text_features

        if self.USE_TEXT == 1 and self.USE_IMAGE == 1:
            features = torch.cat((image_features, text_features), dim=1)
        # cat_size = image_features.size()[1] + text_features.size()[1]

        out = self.classifier(features)

        return out

"Credit to https://gist.github.com/kyamagu/73ab34cbe12f3db807a314019062ad43"
def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    # print(output)
    pred = output >= 0.5
    # print(pred)
    truth = target >= 0.5
    # print(truth)
    acc = pred.eq(truth).sum()
    # print(acc)
    return acc

def validate(dataloader_valid, criterion, device):
    # t = time.time()

    loss = 0
    acc = 0
    i = 0


    for batch in dataloader_valid:

        # print("loadtime, ", time.time() - t)

        image_batch = batch["image"].to(device)
        text_batch = batch["text"].to(device)
        target_batch = batch["class"].to(device)
        target_batch = target_batch.unsqueeze(1)



        with torch.no_grad():

            pred = full_model(image_batch, text_batch)
            acc += accuracy(pred, target_batch)
            size = target_batch.numel()
            loss += criterion(pred, target_batch) * size

            i += target_batch.numel()


    valid_acc = acc.float()/i
    valid_mse = loss/i
    print('acc', acc)
    print('i', i)
    # t = time.time()
    return valid_acc, valid_mse


if __name__ == '__main__':

    HIDDEN_SIZE = 100
    N_EPOCHS = 200
    BATCH_SIZE = 30

    USE_IMAGE = 1
    USE_TEXT = 1

    TRAIN_METADATA_HATE = "hateMemesList.txt.train"
    TRAIN_METADATA_GOOD = "redditMemesList.txt.train"
    VALID_METADATA_HATE = "hateMemesList.txt.valid"
    VALID_METADATA_GOOD = "redditMemesList.txt.valid"
    BASE_PATH = "data/train_data"
    MODEL_SAVE = "models/classifier2.pt"
    # MODEL_SAVE = "models/kk.pt"

    # logname = "H100x4_IF1000v2"
    logname = "H100x4_noDropout_SGD_cropping"
    # logname = "kk"


    start_time = time.time()
    writer = SummaryWriter("logs/" + logname)

    # Configuring CUDA / CPU execution
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    # Get image descriptor
    VGG16_features = torchvision.models.vgg16(pretrained=True)
    VGG16_features.classifier = VGG16_features.classifier[:-3]

    VGG16_features.to(device)

    # To embed text, we use a Pytorch implementation of BERT: Using pythorch BERT implementation from https://github.com/huggingface/pytorch-pretrained-BERT
    # Get Textual Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case="true")
    # Get Textual Embedding.
    bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
    bert_model.to(device)

    # VGG16 output --> 1000 features
    # VGG16 -1 layer --> 4096 features
    # BERT text embedder --> 768 features

    # IMAGE_AND_TEXT_FEATURES = 1768
    IMAGE_FEATURES = 4096
    TEXT_FEATURES = 768

    IMAGE_AND_TEXT_FEATURES = IMAGE_FEATURES * USE_IMAGE + TEXT_FEATURES * USE_TEXT

    full_model = MultimodalClassifier(VGG16_features, bert_model,
                                      IMAGE_AND_TEXT_FEATURES, USE_IMAGE,
                                      USE_TEXT, HIDDEN_SIZE)
    full_model.to(device)

    transform = transforms.Compose([test.Rescale((256, 256)),
                                    test.RandomCrop(224),
                                    test.Tokenize(tokenizer),
                                    test.ToTensor()])

    train_dataset = test.ImagesDataLoader(TRAIN_METADATA_GOOD, TRAIN_METADATA_HATE, BASE_PATH, transform)
    valid_dataset = test.ImagesDataLoader(VALID_METADATA_GOOD, VALID_METADATA_HATE, BASE_PATH, transform)

    DATASET_LEN = train_dataset.__len__()

    dataloader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test.custom_collate)
    dataloader_valid = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test.custom_collate)


    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(full_model.classifier.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.Adam(full_model.classifier.parameters(), lr=0.0001)


    iteration = 0

    for i in range(N_EPOCHS):

        epoch_init = time.time()
        pbar = tqdm(total=DATASET_LEN)
        for batch in dataloader_train:


            image_batch = batch["image"].to(device)
            text_batch = batch["text"].to(device)
            target_batch = batch["class"].to(device)
            target_batch = target_batch.unsqueeze(1)

            optimizer.zero_grad()


            # forward_init = time.time()
            pred = full_model(image_batch, text_batch)
            # forward_final = time.time()
            # print("forward time: ", forward_final - forward_init)

            loss = criterion(pred, target_batch)

            loss.backward()

            optimizer.step()

            writer.add_scalar('train/mse', loss, iteration)
            iteration += 1

            pbar.update(BATCH_SIZE)

            # break

            # print(pred)

        epoch_end = time.time()

        print("Epoch time elapsed:", epoch_end - epoch_init)

        print("Saving full model to " + MODEL_SAVE)
        torch.save(full_model.state_dict(), MODEL_SAVE)
        print("Starting Validation")

        valid_init = time.time()
        valid_acc, valid_loss  = validate(dataloader_valid, criterion, device)
        valid_end = time.time()
        print("Time in validation:", valid_end - valid_init)
        writer.add_scalar('validation/valid_accuracy', valid_acc, i+1)
        writer.add_scalar('validation/valid_mse', valid_loss, i+1)
        # print("Validation accuracy on epoch " + str(i) + ": ")

    end_time = time.time()
    print("Elapsed Time:", end_time - start_time)

    writer.close()

