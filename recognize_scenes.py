# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

# th architecture to use
arch = 'resnet50'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

# load the test image
img_name = '12.jpg'
if not os.access(img_name, os.W_OK):
    img_url = 'http://places.csail.mit.edu/demo/' + img_name
    os.system('wget ' + img_url)

image_folder = Path("data") / "img"
scene_text_folder = Path("info") / "scene"
scene_text_folder.mkdir(exist_ok=True, parents=True)
image_names = list(image_folder.iterdir())

for image_name in tqdm(image_names):
    if not image_name.is_file():
        continue
    img = Image.open(image_name).convert('RGB')
    input_img = V(centre_crop(img).unsqueeze(0))
    text_file = scene_text_folder / image_name.with_suffix(".txt").name
    if text_file.exists():
        continue

    # forward pass
    logit = model.forward(input_img)
    probs = F.softmax(logit, 1).data.squeeze()

    # output the prediction
    with text_file.open('w') as f:
        f.write("\n".join([str(i) for i in probs.tolist()]))

