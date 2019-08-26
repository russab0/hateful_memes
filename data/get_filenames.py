from glob import glob
import cv2

files = glob("untagged_memes1/*") + glob("untagged_memes2/*")

out = open("kkfiles", "w")

for x in files:

    im = cv2.imread(x)

    if im is not None:

        out.write(x + "\n")
