import cv2, numpy as np
from collections import deque

def get_segmentation(source): 

    ret3,th3 = cv2.threshold(source,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ret3,th3 = cv2.threshold(source,0,255,cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(image=th3, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
                                      
    cv2.drawContours(image=th3, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=10, lineType=cv2.LINE_AA)
    # th3 = cv2.adaptiveThreshold(source,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #         cv2.THRESH_BINARY,11,2)
    # blur = cv2.GaussianBlur(source,(5,5),0)
    # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # th3=fill_white_areas(th3)
    
    return th3
