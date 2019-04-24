from random import shuffle

infile = "train_data/redditMemesList.txt"
trainfile = "train_data/redditMemesList.txt.train"
validfile = "train_data/redditMemesList.txt.valid"
TRAIN_PERCENTATGE = 0.85

infile = open(infile)
infile = list(infile)
shuffle(infile)

i = int(TRAIN_PERCENTATGE*len(infile))

train_list = infile[:i]
valid_list = infile[i:]

v = open(validfile, "w")
for x in valid_list:
    v.write(x)
v.close()

t = open(trainfile, "w")
for x in train_list:
    t.write(x)
t.close()