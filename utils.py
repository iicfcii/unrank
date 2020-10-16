import cv2
import numpy as np

def number_to_string(val):
    if val is None: return 'NA'
    return str(val)

def crop(img, rect):
    x, y, w, h = rect

    if x < 0: x = 0
    if y < 0: y = 0
    if x+w > img.shape[1]: w = img.shape[1]-x
    if y+h > img.shape[0]: h = img.shape[0]-y

    img_crop = img[y:y+h,x:x+w]
    return img_crop.copy()

# HSV color space
def match_color(img, lb, ub):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_bin = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)

    if ub[0] > 180: # Red is across 180
        h_match = np.logical_or(img_hsv[:,:,0] > lb[0], img_hsv[:,:,0] < ub[0]-180)
    else:
        h_match = np.logical_and(img_hsv[:,:,0] > lb[0], img_hsv[:,:,0] < ub[0])
    s_match = np.logical_and(img_hsv[:,:,1] > lb[1], img_hsv[:,:,1] < ub[1])
    v_match = np.logical_and(img_hsv[:,:,2] > lb[2], img_hsv[:,:,2] < ub[2])

    img_bin[np.all((h_match, s_match, v_match), axis=0)] = 255

    return img_bin

def read_batch(process, start=0, map='nepal', length=835, num_width=8, num_height=8):
    shape = None
    for i in range(start,int(length/num_width/num_height)+1):
        imgs = []
        for j in range(i*num_width*num_height, i*num_width*num_height+num_width*num_height):
            img = cv2.imread('img/'+map+'/'+map+'_'+str(j*30)+'.jpg', cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros(shape, dtype=np.uint8)
            else:
                print(j*30)
                info, img = process(img)
                shape = img.shape

            if j < length:
                img = cv2.putText(img, info, (0,img.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0))
                img = cv2.putText(img, str(j*30), (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0))
            imgs.append(img)

        imgs_row = []
        for i in range(num_height):
            imgs_row.append(cv2.hconcat(imgs[i*num_width:i*num_width+num_width], num_width))

        img_final = cv2.vconcat(imgs_row, num_height)
        cv2.imshow('read', img_final)
        cv2.waitKey(0)
