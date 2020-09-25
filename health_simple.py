import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

HEALTH_RECT = (165, 580, 90, 70)
HEALTH_NUMBERS_WIDTH = 35
HEALTH_NUMBERS_HEIGHT = 25

SHEAR_X = 0.23
SHEAR_Y = 0.06
SHEAR_X_OFFSET = -int(SHEAR_X*HEALTH_RECT[2])-1
SHEAR_Y_OFFSET = -int(SHEAR_Y*HEALTH_RECT[3])-1
SHEAR_MAT = np.array([[1,SHEAR_X,SHEAR_X_OFFSET],[SHEAR_Y,1,SHEAR_Y_OFFSET]], dtype=np.float32)

def crop(img, rect):
    x, y, w, h = rect

    if x < 0: x = 0
    if y < 0: y = 0
    if x+w > img.shape[1]: w = img.shape[1]-x
    if y+h > img.shape[0]: h = img.shape[0]-y

    img_crop = img[y:y+h,x:x+w]
    return img_crop

NUMBER_INFO = {
    0: (720, (13,22,11,25)),
    1: (2160, (2,21,7,25)),
    2: (720, (2,22,11,25)),
    3: (10890, (20,21,11,25)),
    4: (3690, (9,20,11,25)),
    5: (3630, (9,20,11,25)),
    6: (8400, (19,20,11,25)),
    7: (3690, (19,20,11,25)),
    8: (15450, (9,21,11,25)),
    9: (2160, (20,21,11,25)),
}

def extract_number(number):
    index = NUMBER_INFO[number][0]
    rect = NUMBER_INFO[number][1]

    img = cv2.imread('img/overwatch_1_1_'+str(index)+'.jpg', cv2.IMREAD_COLOR)
    img = crop(img, HEALTH_RECT)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.warpAffine(img, SHEAR_MAT, (img.shape[1]+SHEAR_X_OFFSET,img.shape[0]+SHEAR_Y_OFFSET))
    img = cv2.Canny(img,150,250)
    img = crop(img, rect)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            img = cv2.fillPoly(img, [contour], 0)
        else:
            mask = cv2.fillPoly(img.copy(), [contour], 255)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)), iterations=1, borderValue=0)
    # cv2.imshow('mask', mask)
    # cv2.imshow('number', img)
    # cv2.waitKey(0)

    return img, mask

def extract_numbers():
    templates = {}
    for i in range(10):
        templates[i] = extract_number(i)

    return templates

def process(img):
    img = crop(img, HEALTH_RECT)
    # img = cv2.resize(img, None, fx=RATIO, fy=RATIO)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.warpAffine(img, SHEAR_MAT, (img.shape[1]+SHEAR_X_OFFSET,img.shape[0]+SHEAR_Y_OFFSET))

    edges = cv2.Canny(img,150,250)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,1)), iterations=4, borderValue=0)
    edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1, borderValue=0)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=2, borderValue=0)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        # cv2.rectangle(img, (x,y,w,h), 0)

        if y < 10 or y+h > edges.shape[0]-10:
            cv2.rectangle(edges, (x,y,w,h), 0, thickness=-1)

    # img[edges > 0] = 0

    x,y,w,h = cv2.boundingRect(edges)
    rect_numbers = (x-4, y-2, HEALTH_NUMBERS_WIDTH, HEALTH_NUMBERS_HEIGHT)

    if rect_numbers is not None:
        img = crop(img, rect_numbers)
        if (HEALTH_NUMBERS_HEIGHT,HEALTH_NUMBERS_WIDTH) != img.shape:
            print(img.shape)
            img = np.zeros((HEALTH_NUMBERS_HEIGHT,HEALTH_NUMBERS_WIDTH),dtype=np.uint8)
    else:
        img = np.zeros((HEALTH_NUMBERS_HEIGHT,HEALTH_NUMBERS_WIDTH),dtype=np.uint8)

    return img

def match(img):
    scores = []
    for i in range(10):
        edges = crop(img, (0,0,12,img.shape[0]))
        edges = cv2.Canny(edges,150,250)
        img_number = np.zeros((29,16),dtype=np.uint8)
        img_number[2:2+img.shape[0],2:2+12] = edges
        template, mask = templates[i]
        res = cv2.matchTemplate(img_number, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        res[res == np.inf] = 0
        res[res == -np.inf] = 0

        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)

        scores.append(maxVal)

    number = np.argmax(scores)
    print(number, scores)
    # cv2.imshow('number', img_number)
    # # cv2.imshow('template', template)
    # cv2.waitKey(0)

    return img, number

def process_batch(num_width=15, num_height=8):
    for i in range(1,int(732/num_width/num_height)+1):
        imgs = []
        for j in range(i*num_width*num_height, i*num_width*num_height+num_width*num_height):
            img = cv2.imread('img/overwatch_1_1_'+str(j*30)+'.jpg', cv2.IMREAD_COLOR)
            if img is None:
                # img = np.zeros((HEALTH_NUMBERS_HEIGHT, HEALTH_NUMBERS_WIDTH, 1), dtype=np.uint8)
                img = np.zeros((HEALTH_RECT[3], HEALTH_RECT[2], 1), dtype=np.uint8)
                number_str = 'x'
            else:
                img = process(img)
                img, number = match(img)
                number_str = str(number)
                print(j*30)

            # if j < 732: img = cv2.putText(img, str(j*30), (0,img.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)
            if j < 732: img = cv2.putText(img, number_str, (0,img.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)
            imgs.append(img)

        imgs_row = []
        for i in range(num_height):
            imgs_row.append(cv2.hconcat(imgs[i*num_width:i*num_width+num_width], num_width))

        img_final = cv2.vconcat(imgs_row, num_height)
        cv2.imshow('edges', img_final)
        cv2.waitKey(0)

# template, mask = extract_number(2)
templates = extract_numbers()
process_batch(num_width=10, num_height=8)
