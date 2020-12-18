import cv2
import numpy as np
import matplotlib.pyplot as plt

import utils

MAP_RECT = (500,627,650,60)
MAP_THRESHOLD = 0.8
MAPS = {
    'blizzardworld': 'hybrid',
    'busan': 'control',
    'dorado': 'escort',
    'eichenwalde': 'hybrid',
    'hanamura': 'assult',
    'havana': 'escort',
    'hollywood': 'hybrid',
    'ilios': 'control',
    'junkertown': 'escort',
    'kingsrow': 'hybrid',
    'lijiangtower': 'control',
    'nepal': 'control',
    'numbani': 'hybrid',
    'oasis': 'control',
    'rialto': 'escort',
    'route66': 'escort',
    'anubis': 'assult',
    'volskaya': 'assult',
    'gibraltar': 'escort'
}

def save_templates():
    # 1118
    rect = (678,627,440,60)
    img = cv2.imread('img/anubis/anubis_270.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_anubis.jpg', utils.crop(img, rect))

    rect = (706,627,412,60)
    img = cv2.imread('img/blizzardworld/blizzardworld_300.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_blizzardworld.jpg', utils.crop(img, rect))

    rect = (820,627,298,60)
    img = cv2.imread('img/hanamura/hanamura_240.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_hanamura.jpg', utils.crop(img, rect))

    rect = (782,627,336,60)
    img = cv2.imread('img/junkertown/junkertown_180.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_junkertown.jpg', utils.crop(img, rect))

    rect = (950,627,168,60)
    img = cv2.imread('img/nepal/nepal_300.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_nepal.jpg', utils.crop(img, rect))

    rect = (862,627,256,60)
    img = cv2.imread('img/numbani/numbani_210.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_numbani.jpg', utils.crop(img, rect))

    rect = (914,627,204,60)
    img = cv2.imread('img/rialto/rialto_150.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_rialto.jpg', utils.crop(img, rect))

    rect = (596,627,522,60)
    img = cv2.imread('img/volskaya/volskaya_150.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_volskaya.jpg', utils.crop(img, rect))

    rect = (912,627,206,60)
    img = cv2.imread('img/busan/busan_150.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_busan.jpg', utils.crop(img, rect))

    rect = (912,627,206,60)
    img = cv2.imread('img/busan/busan_150.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_busan.jpg', utils.crop(img, rect))

    rect = (948,627,170,60)
    img = cv2.imread('img/maps_1/maps_1_150.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_ilios.jpg', utils.crop(img, rect))

    rect = (546,627,572,60)
    img = cv2.imread('img/maps_1/maps_1_450.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_gibraltar.jpg', utils.crop(img, rect))

    rect = (546,627,572,60)
    img = cv2.imread('img/maps_1/maps_1_450.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_gibraltar.jpg', utils.crop(img, rect))

    rect = (886,627,232,60)
    img = cv2.imread('img/maps_1/maps_1_1290.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_havana.jpg', utils.crop(img, rect))

    rect = (808,627,310,60)
    img = cv2.imread('img/maps_1/maps_1_1860.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_kingsrow.jpg', utils.crop(img, rect))

    rect = (926,627,192,60)
    img = cv2.imread('img/maps_2/maps_2_180.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_oasis.jpg', utils.crop(img, rect))

    rect = (776,627,342,60)
    img = cv2.imread('img/maps_2/maps_2_510.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_eichenwalde.jpg', utils.crop(img, rect))

    rect = (806,627,312,60)
    img = cv2.imread('img/maps_2/maps_2_960.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_hollywood.jpg', utils.crop(img, rect))

    rect = (750,627,368,60)
    img = cv2.imread('img/maps_3/maps_3_180.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_lijiangtower.jpg', utils.crop(img, rect))

    rect = (854,627,264,60)
    img = cv2.imread('img/maps_3/maps_3_420.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_route66.jpg', utils.crop(img, rect))

    rect = (882,627,236,60)
    img = cv2.imread('img/maps_3/maps_3_600.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/map_dorado.jpg', utils.crop(img, rect))

    # cv2.imshow('img', utils.crop(img, rect))
    # cv2.waitKey(0)

def read_templates():
    templates = {}

    for map in list(MAPS.keys()):
        img = cv2.imread('template/map_'+map+'.jpg', cv2.IMREAD_COLOR)
        if img is None:
            print(map)
            continue
        templates[map] = img

    # cv2.imshow('map', templates['kingsrow'])
    # cv2.waitKey(0)

    return templates

def read_map(src, templates):
    img = utils.crop(src, MAP_RECT)

    scores = []
    for map in templates:
        res = cv2.matchTemplate(img, templates[map], cv2.TM_CCOEFF_NORMED)
        res[np.isnan(res)] = 0
        scores.append((map, np.max(res)))

    scores.sort(reverse=True, key=lambda s:s[1])
    # print(scores)
    score = scores[0]
    if score[1] > MAP_THRESHOLD:
        return score[0]
    else:
        return None

save_templates()
templates = read_templates()
def process_map(src):
    map = read_map(src, templates)

    img = utils.crop(src, MAP_RECT)

    return '{}'.format(
        utils.val_to_string(map),
    ), img
