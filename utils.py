import cv2
import numpy as np

def crop(img, rect):
    x, y, w, h = rect

    if x < 0: x = 0
    if y < 0: y = 0
    if x+w > img.shape[1]: w = img.shape[1]-x
    if y+h > img.shape[0]: h = img.shape[0]-y

    img_crop = img[y:y+h,x:x+w]
    return img_crop

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
