import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from module import test

from pytorch_transformers import *

import time

from tensorboardX import SummaryWriter

from tqdm import tqdm
import numpy as np

from sklearn.metrics import f1_score, accuracy_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


class MultimodalClassifier(nn.Module):
    def __init__(self, image_feat_model, text_feat_model, TOTAL_FEATURES,
                 USE_IMAGE, USE_TEXT, USE_HATE_WORDS, hidden_size, device):

        super(MultimodalClassifier, self).__init__()
        self.im_feat_model = image_feat_model
        self.text_feat_model = text_feat_model

        self.USE_IMAGE = USE_IMAGE
        self.USE_TEXT = USE_TEXT
        self.USE_HATE_WORDS = USE_HATE_WORDS

        self.TOTAL_FEATURES = TOTAL_FEATURES

        self.classifier = nn.Sequential(
            nn.Linear(TOTAL_FEATURES, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, 1),
            # nn.Softmax()
        )

        self.device = device

    def forward(self, image, text, hate_words):

        # with torch.no_grad():
        if self.USE_IMAGE == 1:
            batch_size = image.size()[0]
        elif self.USE_TEXT == 1:
            batch_size = image.size()[0]
        elif self.USE_HATE_WORDS == 1:
            batch_size = image.size()[0]

        features = torch.zeros(batch_size, 1).to(device)

        if self.USE_IMAGE == 1:
            image_features = self.im_feat_model(image)
            features = torch.cat((features, image_features), dim=1)

        if self.USE_TEXT == 1:
            last_hidden_states = self.text_feat_model(text)[0]
            text_features = torch.sum(last_hidden_states, dim=1)
            text_features = text_features / last_hidden_states.size()[1]
            features = torch.cat((features, text_features), dim=1)

        if self.USE_HATE_WORDS == 1:
            features = torch.cat((features, hate_words), dim=1)

        features = features[:, 1:]

        out = self.classifier(features)

        return out


def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    "Credit to https://gist.github.com/kyamagu/73ab34cbe12f3db807a314019062ad43"

    pred = output.flatten() >= 0.5
    truth = target.flatten() >= 0.5
    acc = pred.eq(truth).sum()
    return acc


def validAccuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    "Credit to https://gist.github.com/kyamagu/73ab34cbe12f3db807a314019062ad43"
    pred = output.flatten() >= 0.5
    truth = target.flatten() >= 0.5
    acc = pred.eq(truth)
    return acc


TOP_SIZE = 20
top_losses = []
fewer_losses = []


def getLossFromTuple(item):
    return item[1]


def validate(dataloader_valid, criterion, device, verbose=False):
    loss = 0
    acc = 0
    i = 0
    global top_losses
    global fewer_losses

    top_losses = []
    fewer_losses = []
    all_predictions = []
    all_labels = []

    q = 0
    for batch in dataloader_valid:
        q += 1

        image_batch = batch["image"].to(device)
        text_batch = batch["bert_tokens"].to(device)
        hate_words_batch = batch["hate_words"].to(device)
        paths_batch = batch["image_paths"]

        target_batch = batch["class"].to(device)
        target_batch = target_batch.unsqueeze(1)

        with torch.no_grad():

            pred = full_model(image_batch, text_batch, hate_words_batch)
            all_labels.append(target_batch)
            all_predictions.append(pred)

            distances = (pred - target_batch) ** 2
            kk = validAccuracy(pred, target_batch)
            acc += kk.sum()
            if i != 0:
                print(i, ':', acc / i)

            for j, x in enumerate(distances):
                top_losses.append([paths_batch[j], x])
                fewer_losses.append([paths_batch[j], x])

            # top_losses.sort(key=lambda _: getLossFromTuple(_), reverse=True) TO DO fix
            # fewer_losses.sort(key=lambda _: getLossFromTuple(_), reverse=False) TO DO fix

            top_losses = top_losses[:TOP_SIZE]
            fewer_losses = fewer_losses[:TOP_SIZE]

            size = target_batch.numel()
            loss += float(criterion(pred, target_batch.reshape(-1, 1)) * size)

            i += target_batch.numel()

            # ADDED
            # print(paths_batch)
            # print(all_predictions.cpu().reshape(len(all_predictions), -1)[0], all_labels.cpu().reshape(len(all_labels), -1)[0])
            # all_predictions_copy = (torch.cat(all_predictions.copy()).cpu() >= 0.5).flatten()
            # all_labels_copy = (torch.cat(all_labels.copy()).cpu() >= 0.5).flatten()
            # print(f'Real F1: {f1_score(all_labels_copy, all_predictions_copy)}')
            # print(f'Real Acc: {accuracy_score(all_labels_copy, all_predictions_copy)}')

    valid_acc = acc.float() / i
    valid_mse = loss / i

    all_predictions_copy = (torch.cat(all_predictions).cpu() >= 0.5).flatten()
    all_labels_copy = (torch.cat(all_labels).cpu() >= 0.5).flatten()
    # print(valid_acc)
    print(f'Real F1: {f1_score(all_labels_copy, all_predictions_copy)}')
    print(f'Real Acc: {accuracy_score(all_labels_copy, all_predictions_copy)}')

    return valid_acc, valid_mse


if __name__ == '__main__':

    HIDDEN_SIZE = 50  # TODO was 50
    N_EPOCHS = 15  # TODO was 100
    BATCH_SIZE = 12  # TODO was 25

    UNFREEZE_FEATURES = 999

    USE_IMAGE = 1
    USE_TEXT = 1
    USE_HATE_WORDS = 0

    BASE_PATH = "data/prepared"
    logname = "logs_final_BS25/multimodal3"

    # checkpoint, to_train, MODEL_SAVE = "models/multimodal_HS.pt", True, "models/multimodal_HS.pt"
    # checkpoint, to_train, MODEL_SAVE = "models/classifier.pt.best", True, "models/classifier.pt.best"
    checkpoint, to_train, MODEL_SAVE = None, True, "models/dehate_vgg.pt"
    APPLY_DEHATE_BERT = True
    print(checkpoint, 'to_train', to_train, MODEL_SAVE, 'apply_dehate', APPLY_DEHATE_BERT)

    start_time = time.time()
    writer = SummaryWriter("logs/" + logname)

    # Configuring CUDA / CPU execution
    FORCE_CPU = False
    device = torch.device('cuda:0' if torch.cuda.is_available() and not FORCE_CPU else 'cpu')
    print('device: ', device)

    # Keywords (deprecated)
    hate_list = [
        'ali baba',
        'allah',
        'abbo',
        'black',
        'bomb',
        'dynamite',
        'jew',
        'nazi',
        'niglet',
        'nigger',
        'nigga',
        'paki',
    ]

    # Get image descriptor
    VGG16_features = torchvision.models.vgg16(pretrained=True)
    VGG16_features.classifier = VGG16_features.classifier[:-3]

    VGG16_features.to(device)

    # To embed text, we use a Pytorch implementation of BERT: Using pythorch BERT implementation from https://github.com/huggingface/pytorch-pretrained-BERT
    if APPLY_DEHATE_BERT:
        tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")
        bert_model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")
        bert_model.modules()
        removed = list(bert_model.children())[:-2]  # remove two last layers
        bert_model = torch.nn.Sequential(*removed)
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")

    bert_model.eval()
    bert_model.to(device)

    # IMAGE_AND_TEXT_FEATURES = 1768
    IMAGE_FEATURES = 4096
    TEXT_FEATURES = 768
    HATE_WORDS = len(hate_list)

    IMAGE_AND_TEXT_FEATURES = IMAGE_FEATURES * USE_IMAGE + TEXT_FEATURES * USE_TEXT \
                              + HATE_WORDS * USE_HATE_WORDS

    full_model = MultimodalClassifier(VGG16_features, bert_model,
                                      IMAGE_AND_TEXT_FEATURES, USE_IMAGE,
                                      USE_TEXT, USE_HATE_WORDS, HIDDEN_SIZE,
                                      device)

    if checkpoint is not None:
        full_model.load_state_dict(torch.load(checkpoint, map_location=device))

    full_model.to(device)

    # transform = transforms.Compose([test.Rescale((256, 256)),
    transform = transforms.Compose([test.Rescale((224, 224)),
                                    # test.RandomCrop(224),
                                    test.HateWordsVector(hate_list),
                                    test.Tokenize(tokenizer),
                                    test.ToTensor()])

    transformValid = transforms.Compose([test.Rescale((224, 224)),
                                         # test.RandomCrop(224),
                                         test.HateWordsVector(hate_list),
                                         test.Tokenize(tokenizer),
                                         test.ToTensor()])

    train_dataset = test.ImagesDataLoader('train.jsonl', BASE_PATH, transform)
    valid_dataset = test.ImagesDataLoader('dev.jsonl', BASE_PATH, transformValid)
    # train_dataset = test.ImageTextMatcherDataLoader(TRAIN_METADATA_GOOD, TRAIN_METADATA_HATE, BASE_PATH, transform)
    # valid_dataset = test.ImageTextMatcherDataLoader(VALID_METADATA_GOOD, VALID_METADATA_HATE, BASE_PATH, transformValid)

    DATASET_LEN = train_dataset.__len__()

    dataloader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test.custom_collate)
    dataloader_valid = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test.custom_collate)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    # feature_parameters = list(full_model.im_feat_model.classifier.parameters()) + list(bert_model.parameters())
    #
    # optimizer = torch.optim.SGD(parameters, lr=0.01, momentum=0.9)
    # optimizer = torch.optim.SGD(full_model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(full_model.classifier.parameters())
    # features_optimizer = torch.optim.Adam(feature_parameters)
    features_optimizer = torch.optim.Adam(VGG16_features.classifier.parameters())

    iteration = 0

    best_acc = np.array(-1.00)
    full_model.text_feat_model.train()
    full_model.im_feat_model.train()

    if checkpoint is None or to_train:
        for i in range(N_EPOCHS):

            epoch_init = time.time()
            pbar = tqdm(total=DATASET_LEN)
            for batch in dataloader_train:
                image_batch = batch["image"].to(device)
                text_batch = batch["bert_tokens"].to(device)
                hate_words_batch = batch["hate_words"].to(device)

                target_batch = batch["class"].to(device)

                target_batch = target_batch.unsqueeze(1)

                optimizer.zero_grad()
                features_optimizer.zero_grad()

                pred = full_model(image_batch, text_batch, hate_words_batch)

                # print(pred.shape, pred)
                # print(target_batch.shape, target_batch)
                loss = criterion(pred, target_batch.reshape(-1, 1))

                loss.backward()

                optimizer.step()

                if i >= UNFREEZE_FEATURES:
                    features_optimizer.step()

                writer.add_scalar('train/mse', loss, iteration * BATCH_SIZE)
                iteration += 1

                pbar.update(BATCH_SIZE)

            epoch_end = time.time()

            print("Epoch time elapsed:", epoch_end - epoch_init)

            print("Starting Validation")

            valid_init = time.time()

            full_model.eval()
            full_model.text_feat_model.eval()
            full_model.im_feat_model.eval()
            valid_acc, valid_loss = validate(dataloader_valid, criterion, device)
            full_model.text_feat_model.train()
            full_model.im_feat_model.train()
            full_model.train()

            valid_acc_np = valid_acc.cpu().numpy()

            if valid_acc_np > best_acc:
                print("Saving full model to " + MODEL_SAVE + ".best")
                torch.save(full_model.state_dict(), MODEL_SAVE + '.best')
                logfile = open(MODEL_SAVE + ".best.log", "w")
                best_acc = valid_acc_np
                logfile.write("best_acc:" + str(valid_acc_np) + "\n" + "best epoch: " + str(i) + "\n")
                logfile.close()
                best_acc = valid_acc

                accs = open("results/accuracys", "w")

                accs.write("Smaller losses from best acc (epoch : " + str(i) + ")\n")

                for x in fewer_losses:
                    accs.write(str(x[0]) + "\t" + str(x[1]) + "\n")

                accs.write("Top Loss:\n")
                for x in top_losses:
                    accs.write(str(x[0]) + "\t" + str(x[1]) + "\n")

                accs.close()

            valid_end = time.time()
            print("Time in validation:", valid_end - valid_init)
            writer.add_scalar('validation/valid_accuracy', valid_acc, i + 1)
            writer.add_scalar('validation/valid_mse', valid_loss, i + 1)
    else:
        full_model.eval()
        full_model.text_feat_model.eval()
        full_model.im_feat_model.eval()
        valid_acc, valid_loss = validate(dataloader_valid, criterion, device)

    end_time = time.time()
    print("Elapsed Time:", end_time - start_time)

    # accs.close()

    writer.close()
