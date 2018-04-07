'''
Script to resize all images in a folder specified as arg1
to a square of size specified in arg2
Creates a folder "resized" at the same level as the folder in arg1
Grabs all .jpg and .png files in the source
'''
import sys
import os
from utils.dataproc import read_and_resize_image, write_img

if __name__=="__main__":
    src = sys.argv[1]
    val = int(sys.argv[2])
    newsize = (val,val)

    # clear out the last character if it is a /
    if src.endswith("/"):
        src = src[:-1]

    # create a directory at the same level as the last directory of the source
    target = os.path.join(os.path.split(src)[0], "resized%d"%val)
    userchoice = raw_input("Source is %s\nTarget will be %s\nNew size will be%s\nProceed? (y/n) "
                           %(src, target, str(newsize)))

    os.mkdir(target)

    if userchoice=="y":
        allimgfiles = filter(lambda x: x.endswith("png") or x.endswith("jpg"), os.listdir(src))
        for f in allimgfiles:
            write_img(os.path.join(target,f), read_and_resize_image(os.path.join(src, f), newsize))
