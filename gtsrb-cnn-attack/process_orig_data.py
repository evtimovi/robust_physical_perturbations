'''
Script to crop (only crop, no resizing)
all images in a specified folder (presumably the original dataset) [arg 1]
and save them to a given target folder [arg 2]
Assumes folder org is datasettop/class/images
'''
import sys
from utils.dataproc import process_orig_data

if __name__=="__main__":
    origsetpath = sys.argv[1]
    targetdir = sys.argv[2]
    userchoice = raw_input("Source dir %s\n Target dir %s\nProceed (y/n)?"%(origsetpath, targetdir))
    if userchoice == "y":
        print "Okay, processing"
        process_orig_data(origsetpath, targetdir)
