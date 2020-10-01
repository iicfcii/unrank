import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

HERO_RECT_1 = (75,48,21,25)
HERO_RECT_7 = (864,48,21,25)
STAT_RECT_X_OFFSET = 71
MATCH_PADDING = 10

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

def crop(img, rect):
    x, y, w, h = rect

    if x < 0: x = 0
    if y < 0: y = 0
    if x+w > img.shape[1]: w = img.shape[1]-x
    if y+h > img.shape[0]: h = img.shape[0]-y

    img_crop = img[y:y+h,x:x+w]
    return img_crop

def read_hero_rects():
    hero_rects = {}

    for i in range(1,7):
        hero_rects[i] = (int(HERO_RECT_1[0]+(i-1)*STAT_RECT_X_OFFSET),
                             HERO_RECT_1[1],
                             HERO_RECT_1[2],
                             HERO_RECT_1[3])

    for i in range(7,13):
        hero_rects[i] = (int(HERO_RECT_7[0]+(i-7)*STAT_RECT_X_OFFSET),
                             HERO_RECT_7[1],
                             HERO_RECT_7[2],
                             HERO_RECT_7[3])

    # img = cv2.imread('img/overwatch_1_1_1860.jpg', cv2.IMREAD_COLOR) # 3030
    # for i in range(1,13):
    #     img_mark = cv2.rectangle(img, hero_rects[i], (255,255,255), thickness=1)
    # cv2.imshow('img', img_mark)
    # cv2.waitKey(0)

    return hero_rects

def save_hero_templates():
    hero_rects = read_hero_rects()
    img = cv2.imread('img/overwatch_1_1_1860.jpg', cv2.IMREAD_COLOR)
    img_2 = cv2.imread('img/overwatch_1_1_3900.jpg', cv2.IMREAD_COLOR)
    img_3 = cv2.imread('img/overwatch_1_1_7710.jpg', cv2.IMREAD_COLOR)
    img_4 = cv2.imread('img/overwatch_1_1_14340.jpg', cv2.IMREAD_COLOR)
    img_5 = cv2.imread('img/overwatch_1_1_21150.jpg', cv2.IMREAD_COLOR)
    img_6 = cv2.imread('img/overwatch_1_1_300.jpg', cv2.IMREAD_COLOR)

    cv2.imwrite('template/hero_moira.jpg', crop(img, hero_rects[1]))
    cv2.imwrite('template/hero_dva.jpg', crop(img, hero_rects[2]))
    cv2.imwrite('template/hero_junkrat.jpg', crop(img, (hero_rects[3][0]-3,hero_rects[3][1],hero_rects[3][2],hero_rects[3][3])))
    cv2.imwrite('template/hero_roadhog.jpg', crop(img, hero_rects[4]))
    cv2.imwrite('template/hero_ashe.jpg', crop(img, hero_rects[8]))
    cv2.imwrite('template/hero_mercy.jpg', crop(img, (hero_rects[6][0]-3,hero_rects[6][1],hero_rects[6][2],hero_rects[6][3])))
    cv2.imwrite('template/hero_doomfist.jpg', crop(img, (hero_rects[7][0]-2,hero_rects[7][1],hero_rects[7][2],hero_rects[7][3])))
    cv2.imwrite('template/hero_wreckingball.jpg', crop(img, hero_rects[10]))
    cv2.imwrite('template/hero_ana.jpg', crop(img, hero_rects[11]))

    cv2.imwrite('template/hero_soldier76.jpg', crop(img_2, (hero_rects[3][0]-3,hero_rects[3][1],hero_rects[3][2],hero_rects[3][3])))

    cv2.imwrite('template/hero_zarya.jpg', crop(img_3, (hero_rects[2][0]-3,hero_rects[2][1],hero_rects[2][2],hero_rects[2][3])))
    cv2.imwrite('template/hero_mei.jpg', crop(img_3, hero_rects[3]))
    cv2.imwrite('template/hero_reinhardt.jpg', crop(img_3, (hero_rects[12][0]+3,hero_rects[12][1],hero_rects[12][2],hero_rects[12][3])))
    cv2.imwrite('template/hero_baptiste.jpg', crop(img_3, (hero_rects[9][0]-6,hero_rects[9][1],hero_rects[9][2],hero_rects[9][3])))

    cv2.imwrite('template/hero_mccree.jpg', crop(img_4, (hero_rects[3][0]+2,hero_rects[3][1],hero_rects[3][2],hero_rects[3][3])))
    cv2.imwrite('template/hero_widowmaker.jpg', crop(img_4, (hero_rects[5][0]-4,hero_rects[5][1],hero_rects[5][2],hero_rects[5][3])))

    cv2.imwrite('template/hero_genji.jpg', crop(img_5, (hero_rects[5][0],hero_rects[5][1],hero_rects[5][2],hero_rects[5][3])))
    cv2.imwrite('template/hero_tracer.jpg', crop(img_5, (hero_rects[8][0]+6,hero_rects[8][1],hero_rects[8][2],hero_rects[8][3])))

    cv2.imwrite('template/hero_unknown1.jpg', crop(img_6, hero_rects[1]))
    cv2.imwrite('template/hero_unknown2.jpg', crop(img_6, hero_rects[8]))

def read_hero_templates():
    templates = {}

    for hero in HEROES:
        templates[hero] = cv2.imread('template/hero_'+hero+'.jpg', cv2.IMREAD_COLOR)

    return templates

def read_hero(img, hero_rect, templates):
    hero_rect = (hero_rect[0]-MATCH_PADDING,
                 hero_rect[1]-MATCH_PADDING,
                 hero_rect[2]+2*MATCH_PADDING,
                 hero_rect[3]+2*MATCH_PADDING)

    img = crop(img, hero_rect)

    match = {}
    for hero in templates:
        res = cv2.matchTemplate(img, templates[hero], cv2.TM_CCOEFF_NORMED)
        score = np.max(res)

        match[hero] = score

    # print(match)
    hero = max(match, key=match.get) # Hero with max score
    max_score = match[hero]
    if max_score < 0.8:
        hero = 'none'

    return hero, img

def read_batch(num_width=40, num_height=20):
    templates = read_hero_templates()
    hero_rects = read_hero_rects()

    for i in range(0,int(732/num_width/num_height)+1):
        imgs = []
        for j in range(i*num_width*num_height, i*num_width*num_height+num_width*num_height):
            img = cv2.imread('img/overwatch_1_1_'+str(j*30)+'.jpg', cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((HERO_RECT_1[3]+MATCH_PADDING*2, HERO_RECT_1[2]+MATCH_PADDING*2, 3), dtype=np.uint8)
                hero = ''
            else:
                hero, img = read_hero(img, hero_rects[1], templates)


            if j < 732: img = cv2.putText(img, str(hero), (0,img.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
            imgs.append(img)

        imgs_row = []
        for i in range(num_height):
            imgs_row.append(cv2.hconcat(imgs[i*num_width:i*num_width+num_width], num_width))

        img_final = cv2.vconcat(imgs_row, num_height)
        cv2.imshow('read', img_final)
        cv2.waitKey(0)

# hero_rects = read_hero_rects()
# save_hero_templates()
# templates = read_hero_templates()
# img = cv2.imread('img/overwatch_1_1_11220.jpg', cv2.IMREAD_COLOR)
# print(read_hero(img, hero_rects[6], templates))

read_batch()
