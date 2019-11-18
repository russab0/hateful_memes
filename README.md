## memes_processing

Deep Learning Applied to Hate Meme Detection

Multimodal classification (text + image)

### Usage

#### Instalation
The main dependencies for the project are Pytorch (visit pytorch.org for installation), torchvision, OpenCV, numpy, tqdm, tensorflow and tensorboardX.
Tensorflow is only to support plotting curves with tensorboard.

You will also need BERT implementations from https://github.com/huggingface/transformers.

Since many of the repos are rapidly changing, we include a txt file with the result of the `pip freeze` command, which includes all the libraries installed and their version.

#### Data preparation

The data from the paper is not provided, however, you can find some utility script inside `./data/` folder that download hate class memes.

The data format requires 4 metadata files. One for each class (hate-nohate) and susbset (validation). Each line of this files represents a data sample. 
Each line will be a path to the image data. 
The system will also look for the extistence of the `"path_to_image".ocr` file, which shoud be a txt file with the precomputed ocr extraction.
Notice that not extracting the OCR previously can take extensive time and can slow the inference and training process up to 2000%.
All the paths in this file must be relative to the `BASE_PATH` parameter in the code, set to default to `./data/train_data/`

#### Training
To train the model simply run

```python train.py```

All the training parameters are hardcoded inside this scripts.
A brief description of the most important parameters are:

 - HIDDEN_SIZE. Number of hidden neurons in the MLP. 
 - N_EPOCHS. Number of training epochs.
 - BATCH_SIZE. Training batch size

- UNFREEZE_FEATURES. Experimental feature for unfreezing the VGG and BERT weights after a certain number of epochs.

- USE_IMAGE. Inlcude VGG image description in the model (1, 0)
- USE_TEXT. Inlcude BERT text description in the model (1, 0)
- USE_HATE_WORDS. Experimental feature to also concatenate a one-hot encoding of the presence of certain hate keywords. Deprecated.

- TRAIN_METADATA_HATE. Training hate class metadata file.
- TRAIN_METADATA_GOOD. Training non-hate class metadata file.
- VALID_METADATA_HATE. Validation hate class metadata file.
- VALID_METADATA_GOOD. Validation non-hate class metadata file.
- BASE_PATH. Explained in Data Preparation.

- MODEL_SAVE. Path where the best model will be saved.

- logname. Path where the tensorboardX logs will be saved.

- checkpoint. Path to model from which to start training. Assing `None` if you want to train from scratch.


#### Testing
The input format is the same as training. It will output a text file that contains the image path, the ground truth class and the prediction of the image, separated by spaces. 
One example per line. 

You can download a pretrained model from [here](https://imatge.upc.edu/web/sites/default/files/projects/language/public_html/2019-neuripsws-hatespeechdetection/multimodal_HS.pt):

