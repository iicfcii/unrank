import cv2
import numpy as np
import matplotlib.pyplot as plt

NUMBER_IMAGE_INDICES = (2340, 2190, 2220, 2580, 3000, 2610, 3210, 3300, 3450, 3570)

STAT_RECT_2 = (895,50,80,50) # 895 305 967
STAT_RECT_1 = (305,50,80,50) # 895 305 967
STAT_ULT_LEFT = 913
STAT_ULT_TOP = 55
STAT_ULT_WIDTH = 6
STAT_ULT_HEIGHT = 15
STAT_ULT_TILT = 4
STAT_ULT_DIGIT_1_MASK_1_X_OFFSET = 2
STAT_ULT_DIGIT_1_MASK_7_X_OFFSET = 1
STAT_ULT_DIGIT_1_MASK_POLY = (STAT_ULT_LEFT,STAT_ULT_TOP,
                              STAT_ULT_LEFT+STAT_ULT_WIDTH,STAT_ULT_TOP,
                              STAT_ULT_LEFT+STAT_ULT_WIDTH-STAT_ULT_TILT,STAT_ULT_TOP+STAT_ULT_HEIGHT,
                              STAT_ULT_LEFT-STAT_ULT_TILT,STAT_ULT_TOP+STAT_ULT_HEIGHT)
mask_poly = np.array(STAT_ULT_DIGIT_1_MASK_POLY, dtype=np.float32).reshape((-1,2))
STAT_ULT_DIGIT_1_RECT = cv2.boundingRect(mask_poly)

STAT_ULT_NUMBER_TOO_CLOSE_THRESHOLD = 5
THRESHOLD = 0.9

def crop(img, rect):
    img_crop = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    return img_crop

def save_templates():
    for number, index in enumerate(NUMBER_IMAGE_INDICES):
        img = cv2.imread('img/overwatch_1_1_'+str(index)+'.jpg', cv2.IMREAD_COLOR)
        img_mark = cv2.polylines(img.copy(), [mask_poly.astype(np.int32)], True, (255,255,255), thickness=1)
        # cv2.imshow('img', img_mark)
        # cv2.waitKey(0)
        cv2.imwrite('template/ult_number_'+str(number)+'.jpg', crop(img, STAT_ULT_DIGIT_1_RECT))


def read_template(number):
    img = cv2.imread('template/ult_number_'+str(number)+'.jpg', cv2.IMREAD_GRAYSCALE) # IMREAD_COLOR
    mask_poly_offset = mask_poly-np.tile(STAT_ULT_DIGIT_1_RECT[0:2],(4,1))

    # Adjust mask a bit for 1 and 7
    if number == 1:
        mask_poly_offset[0,0] = mask_poly_offset[0,0]+STAT_ULT_DIGIT_1_MASK_1_X_OFFSET
        mask_poly_offset[3,0] = mask_poly_offset[3,0]+STAT_ULT_DIGIT_1_MASK_1_X_OFFSET

    if number == 7:
        mask_poly_offset[0,0] = mask_poly_offset[0,0]+STAT_ULT_DIGIT_1_MASK_7_X_OFFSET
        mask_poly_offset[3,0] = mask_poly_offset[3,0]+STAT_ULT_DIGIT_1_MASK_7_X_OFFSET

    img_mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    cv2.fillConvexPoly(img_mask, mask_poly_offset.astype(np.int32), (255,255,255))

    return (img, img_mask)

def read_templates():
    templates = {}
    for number in range(10):
        template, mask = read_template(number)
        template[mask == 0] = 0
        templates[number] = (template, mask)

        # cv2.imshow(str(number), template)
        # cv2.waitKey(0)

    return templates

# save_templates()

templates = read_templates()

def read_ult(stat_rect):
    data_ult = []

    for i in range(732):
        index = i*30
        # index = 10890
        img = cv2.imread('img/overwatch_1_1_'+str(index)+'.jpg', cv2.IMREAD_COLOR)
        img_cropped = crop(img, stat_rect)
        img_cropped_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        img_cropped_gray = cv2.resize(img_cropped_gray, None, fx=2, fy=2)

        match = []
        for number in range(10):
            template, mask = templates[number]
            template = cv2.resize(template, None, fx=2, fy=2)
            mask = cv2.resize(mask, None, fx=2, fy=2)
            # cv2.imshow('template', template)
            # cv2.imshow('mask', mask)
            # cv2.waitKey(0)

            res = cv2.matchTemplate(img_cropped_gray, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            locations = np.where(res>=THRESHOLD)

            if len(locations[0]) == 0: continue

            # Clean up clustered location
            locations_simple = []
            for l in zip(locations[1], locations[0]):
                tooClose = False
                for l_simple in locations_simple:
                    if (l[0]-l_simple[0])**2+(l[1]-l_simple[1])**2 < STAT_ULT_NUMBER_TOO_CLOSE_THRESHOLD **2:
                        tooClose = True
                        break
                if not tooClose:
                    locations_simple.append(l)
                    match.append((number,l, res[l[1],l[0]])) # Consider as two numbers

        if len(match) > 2:
            # Only save two matches with highest res
            match.sort(reverse=True, key=lambda l:l[2])
            match = match[0:2]

        for i in range(len(match)-1,-1,-1):
            # Remove match with inf res
            if match[i][2] == float('inf'):
                match.pop(i)

        # Sort according to x so that value is correct
        match.sort(key=lambda l:l[1][0])

        if len(match) == 2:
            ult = match[0][0]*10+match[1][0]
        elif len(match) == 1:
            ult = match[0][0]
        else:
            ult = -1

        data_ult.append(ult)

        # print(ult, match)
        # cv2.imshow('match', img_cropped)
        # cv2.waitKey(0)

    return data_ult

data_ult_1 = read_ult(STAT_RECT_1)
data_ult_2 = read_ult(STAT_RECT_2)

plt.plot(data_ult_1, label='Ashe 1')
plt.plot(data_ult_2, label='Ashe 2')
plt.ylabel('ult charge')
plt.legend()
plt.show()
