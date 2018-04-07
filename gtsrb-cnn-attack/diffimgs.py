import numpy as np
import cv2
import sys

if __name__=="__main__":
    img1 = cv2.resize(cv2.imread(sys.argv[1]), (32, 32))
    img2 = cv2.resize(cv2.imread(sys.argv[2]), (32, 32))
    diff = img1-img2
    cv2.imwrite("diff_img.png", diff)
    cv2.imwrite("img1.png", img1)
    cv2.imwrite("img2.png", img2)
