# -*- encoding: utf-8 -*-
import numpy as np
from PIL import Image
import cv2
from imutils import paths
import argparse

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

if __name__== '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, help="设置输入图片路径")
    ap.add_argument("-t", "--threshold", type=float, default=100.0, help="设置模糊阈值")
    args= vars(ap.parse_args())

    for imagePath in paths.list_images(args["images"]):
        image= cv2.imread(imagePath)
        gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm= variance_of_laplacian(gray)
        text= "not blurry"

        if fm<args["threshold"]:
            text= "blurry"

        cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
        cv2.imshow("image", image)
        key= cv2.waitKey(0)



