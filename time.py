import cv2
import numpy as np

TIME_RECT = (735,35,32,16)
TIME_DIGIT_0_RECT = (755,37,7,11)
THRESHOLD = 0.8

def img_crop(img, rect):
    img_crop = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    return img_crop

def save_sample_time():
    START = 1800
    STEP = 30
    for i in range(10):
        img = cv2.imread('img/overwatch_1_1_'+str(START+STEP*i)+'.jpg')
        img_time = img[TIME_RECT[1]:TIME_RECT[1]+TIME_RECT[3],TIME_RECT[0]:TIME_RECT[0]+TIME_RECT[2]]
        img_time_digit_0 = img[TIME_DIGIT_0_RECT[1]:TIME_DIGIT_0_RECT[1]+TIME_DIGIT_0_RECT[3],TIME_DIGIT_0_RECT[0]:TIME_DIGIT_0_RECT[0]+TIME_DIGIT_0_RECT[2]]

        cv2.imshow('time', img_time_digit_0)
        # digit_N_time
        cv2.imwrite('sample/digit_'+str(9-i)+'_time.jpg', img_time_digit_0);
        cv2.waitKey(0)

# save_sample_time()

def detec_digit_time(img, rect):
    img_time = img_crop(img, rect)
    cv2.imshow('crop', img_time)
    match = []
    for i in range(10):
        img_digit = cv2.imread('sample/digit_'+str(i)+'_time.jpg', cv2.IMREAD_GRAYSCALE)
        res = cv2.matchTemplate(img_time,img_digit,cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= THRESHOLD)
        print(loc)
    #     match.append(res[0,0])
    #
    # print(np.argmax(match), match)
    # if len(match) == 1:
    #     return match[0]
    # else:
    #     print(match)
    #     assert(len(match) == 0)
    #     return None

img = cv2.imread('img/overwatch_1_1_4380.jpg', cv2.IMREAD_GRAYSCALE)
print(detec_digit_time(img, TIME_RECT))

cv2.imshow('time', img)
cv2.waitKey(0)
