import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

HEROES = [
    'ana',
    'ashe',
    'baptiste',
    'doomfist',
    'dva',
    'genji',
    'junkrat',
    'mccree',
    'mei',
    'mercy',
    'moira',
    'reinhardt',
    'roadhog',
    'soldier76',
    'tracer',
    'widowmaker',
    'wreckingball',
    'zarya',
]

ELIMS_RECT = (910,110,350,175)

RATIO = 1.5 # Scale image to make threshold clearer
MATCH_THRESHOLD = 0.9


def crop(img, rect):
    x, y, w, h = rect

    if x < 0: x = 0
    if y < 0: y = 0
    if x+w > img.shape[1]: w = img.shape[1]-x
    if y+h > img.shape[0]: h = img.shape[0]-y

    img_crop = img[y:y+h,x:x+w]
    return img_crop

def save_elim_templates():
    w = 20
    h = 20

    img = cv2.imread('img/overwatch_1_1_2670.jpg', cv2.IMREAD_COLOR)
    rect = (1142,117,w,h)
    cv2.imwrite('template/elim_moira.jpg', crop(img, rect))
    rect = (1046,117,w,h)
    cv2.imwrite('template/elim_doomfist.jpg', crop(img, rect))

    img = cv2.imread('img/overwatch_1_1_2940.jpg', cv2.IMREAD_COLOR)
    rect = (1119,117,w,h)
    cv2.imwrite('template/elim_ashe.jpg', crop(img, rect))
    rect = (1085,152,w,h)
    cv2.imwrite('template/elim_wreckingball.jpg', crop(img, rect))
    rect = (1178,152,w,h)
    cv2.imwrite('template/elim_mercy.jpg', crop(img, rect))
    rect = (1151,186,w,h)
    cv2.imwrite('template/elim_junkrat.jpg', crop(img, rect))
    rect = (1200,221,w,h)
    cv2.imwrite('template/elim_roadhog.jpg', crop(img, rect))

    img_mark = cv2.rectangle(img.copy(), rect, (255,255,255), thickness=1)
    cv2.imshow('img', img_mark)
    cv2.waitKey(0)

def read_elim_templates():
    templates = {}
    heros = [
        'moira',
        'doomfist',
        'ashe',
        'wreckingball',
        'mercy',
        'junkrat',
        'roadhog',
    ]

    for hero in heros:
        img = cv2.imread('template/elim_'+hero+'.jpg', cv2.IMREAD_COLOR)
        templates[hero] = img

    return templates

def read_elim_from_hero_templates():
    templates = {}
    ratio = 0.7 # Scale the hero template to match the size of elim icon

    for hero in HEROES:
        img = cv2.imread('template/hero_'+hero+'.jpg', cv2.IMREAD_COLOR)
        img = cv2.resize(img, None, fx=ratio, fy=ratio)
        templates[hero] = cv2.resize(img, None, fx=RATIO, fy=RATIO)

    return templates


def extract_location(res):
    locations = np.where(res>=MATCH_THRESHOLD)

    if len(locations[0]) == 0: return []

    # Sort locations so that ones with higher res are preserved
    # locations:[(x,y,score),(x,y,score),(x,y,score)...]
    locations = [(locations[1][i], locations[0][i], res[locations[0][i],locations[1][i]]) for i in range(len(locations[0]))]
    locations.sort(reverse=True, key=lambda l:l[2])

    # Clean up clustered location
    locations_simple = []
    for l in locations:
        tooClose = False
        for l_simple in locations_simple:
            if (l[0]-l_simple[0])**2+(l[1]-l_simple[1])**2 < 30 **2:
                tooClose = True
                break
        if not tooClose:
            locations_simple.append(l)

    # print(locations_simple)

    return locations_simple

def read_elim(img, templates):
    t0 = time.time()

    img = crop(img, ELIMS_RECT)
    img = cv2.resize(img, None, fx=RATIO, fy=RATIO)

    h, w = templates['moira'].shape[0:2]

    for hero in templates:
        res = cv2.matchTemplate(img, templates[hero], cv2.TM_CCOEFF_NORMED)
        locations = extract_location(res)
        for l in locations:
            print(l)
            img = cv2.rectangle(img, (l[0],l[1],w,h), (255,255,255), thickness=2)

    t1 = time.time()
    # print(t1-t0)

    img = cv2.resize(img, None, fx=1/RATIO, fy=1/RATIO)
    return img

def read_batch(num_width=5, num_height=5):
    templates = read_elim_from_hero_templates()

    for i in range(28,int(732/num_width/num_height)+1):
        imgs = []
        for j in range(i*num_width*num_height, i*num_width*num_height+num_width*num_height):
            img = cv2.imread('img/overwatch_1_1_'+str(j*30)+'.jpg', cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((ELIMS_RECT[3], ELIMS_RECT[2], 3), dtype=np.uint8)
                elim = 0
            else:
                print(j*30)
                img = read_elim(img, templates)


            if j < 732: img = cv2.putText(img, str(j*30), (0,img.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            imgs.append(img)

        imgs_row = []
        for i in range(num_height):
            imgs_row.append(cv2.hconcat(imgs[i*num_width:i*num_width+num_width], num_width))

        img_final = cv2.vconcat(imgs_row, num_height)
        cv2.imshow('read', img_final)
        cv2.waitKey(0)

# save_elim_templates()
# templates = read_elim_templates()

# img = cv2.imread('img/overwatch_1_1_20700.jpg', cv2.IMREAD_COLOR)
# img = read_elim(img, templates)

read_batch()
