import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

import utils

HERO_RECT_LEFT = (40,48,410,25)
HERO_RECT_RIGHT = (835,48,410,25)
HERO_RECT_1 = (75,48,15,25) # Cant be too wide otherwise discord will affect the match
HERO_RECT_7 = (864,48,15,25)
STAT_RECT_X_OFFSET = 71
MATCH_PADDING = 10

HERO_THRESHOLD = 0.7

HEROES = [
    'ana',
    'ashe',
    'baptiste',
    'bastion',
    'brigitte',
    'doomfist',
    'dva',
    'echo',
    'genji',
    'hanzo',
    'junkrat',
    'lucio',
    'mccree',
    'mei',
    'mercy',
    'moira',
    'orisa',
    'pharah',
    'reaper',
    'reinhardt',
    'roadhog',
    'sigma',
    'soldier76',
    'sombra',
    'symmetra',
    'torbjorn',
    'tracer',
    'widowmaker',
    'winston',
    'wreckingball',
    'zarya',
    'zenyatta',
]

def offset_rect(rect, dx, dy):
    rect = list(rect)
    rect[0] += dx
    rect[1] += dy
    return tuple(rect)

def read_rects():
    rects = {}

    for i in range(1,7):
        rects[i] = (int(HERO_RECT_1[0]+(i-1)*STAT_RECT_X_OFFSET),
                        HERO_RECT_1[1],
                        HERO_RECT_1[2],
                        HERO_RECT_1[3])

    for i in range(7,13):
        rects[i] = (int(HERO_RECT_7[0]+(i-7)*STAT_RECT_X_OFFSET),
                        HERO_RECT_7[1],
                        HERO_RECT_7[2],
                        HERO_RECT_7[3])

    # img = cv2.imread('img/volskaya/volskaya_1860.jpg', cv2.IMREAD_COLOR) # 3030
    # for i in range(1,13):
    #     img_mark = cv2.rectangle(img, rects[i], (255,255,255), thickness=1)
    # cv2.imshow('img', img_mark)
    # cv2.waitKey(0)

    return rects

def save_templates():
    rects = read_rects()

    img_1 = cv2.imread('img/volskaya/volskaya_1860.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hero_moira.jpg', utils.crop(img_1, rects[1]))
    cv2.imwrite('template/hero_dva.jpg', utils.crop(img_1, rects[2]))
    cv2.imwrite('template/hero_junkrat.jpg', utils.crop(img_1, offset_rect(rects[3], -3, 0)))
    cv2.imwrite('template/hero_roadhog.jpg', utils.crop(img_1, rects[4]))
    cv2.imwrite('template/hero_ashe.jpg', utils.crop(img_1, rects[8]))
    cv2.imwrite('template/hero_mercy.jpg', utils.crop(img_1, offset_rect(rects[6], -3, 0)))
    cv2.imwrite('template/hero_doomfist.jpg', utils.crop(img_1, offset_rect(rects[7], -2, 0)))
    cv2.imwrite('template/hero_wreckingball.jpg', utils.crop(img_1, rects[10]))
    cv2.imwrite('template/hero_ana.jpg', utils.crop(img_1, offset_rect(rects[11], -4, 0)))

    img_2 = cv2.imread('img/volskaya/volskaya_3900.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hero_soldier76.jpg', utils.crop(img_2, offset_rect(rects[3], -3, 0)))

    img_3 = cv2.imread('img/volskaya/volskaya_7710.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hero_zarya.jpg', utils.crop(img_3, offset_rect(rects[2], -3, 0)))
    cv2.imwrite('template/hero_mei.jpg', utils.crop(img_3, rects[3]))
    cv2.imwrite('template/hero_reinhardt.jpg', utils.crop(img_3, offset_rect(rects[12], 3, 0)))
    cv2.imwrite('template/hero_baptiste.jpg', utils.crop(img_3, offset_rect(rects[9], -6, 0)))

    img_4 = cv2.imread('img/volskaya/volskaya_14340.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hero_mccree.jpg', utils.crop(img_4, offset_rect(rects[3], 2, 0)))
    cv2.imwrite('template/hero_widowmaker.jpg', utils.crop(img_4, offset_rect(rects[5], -4, 0)))

    img_5 = cv2.imread('img/volskaya/volskaya_21150.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hero_genji.jpg', utils.crop(img_5, rects[5]))
    cv2.imwrite('template/hero_tracer.jpg', utils.crop(img_5, offset_rect(rects[8], 6, 0)))

    img_7 = cv2.imread('img/rialto/rialto_1260.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hero_hanzo.jpg', utils.crop(img_7, offset_rect(rects[7], 8, 0)))
    cv2.imwrite('template/hero_sigma.jpg', utils.crop(img_7, offset_rect(rects[10], 6, 0)))
    cv2.imwrite('template/hero_echo.jpg', utils.crop(img_7, offset_rect(rects[12], 4, 0)))

    img_8 = cv2.imread('img/hanamura/hanamura_1260.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hero_sombra.jpg', utils.crop(img_8, rects[1]))
    cv2.imwrite('template/hero_zenyatta.jpg', utils.crop(img_8, rects[3]))
    cv2.imwrite('template/hero_reaper.jpg', utils.crop(img_8, offset_rect(rects[9], 2, 0)))

    img_9 = cv2.imread('img/hanamura/hanamura_38910.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hero_symmetra.jpg', utils.crop(img_9, offset_rect(rects[7], 4, 0)))

    img_10 = cv2.imread('img/numbani/numbani_3030.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hero_torbjorn.jpg', utils.crop(img_10, offset_rect(rects[1], 2, 0)))
    cv2.imwrite('template/hero_lucio.jpg', utils.crop(img_10, rects[9]))

    img_11 = cv2.imread('img/numbani/numbani_16200.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hero_winston.jpg', utils.crop(img_11, rects[4]))
    cv2.imwrite('template/hero_orisa.jpg', utils.crop(img_11, rects[12]))

    img_12 = cv2.imread('img/nepal/nepal_16530.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hero_pharah.jpg', utils.crop(img_12, offset_rect(rects[1], 2, 0)))

    img_13 = cv2.imread('img/nepal/nepal_20700.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hero_brigitte.jpg', utils.crop(img_13, rects[5]))

    img_14 = cv2.imread('img/anubis/anubis_9390.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/hero_bastion.jpg', utils.crop(img_14, offset_rect(rects[5], -4, 0)))

def read_templates():
    templates = {}

    for hero in HEROES:
        templates[hero] = cv2.imread('template/hero_'+hero+'.jpg', cv2.IMREAD_COLOR)

    return templates

def read_hero(src, rect, templates):
    rect = (rect[0]-MATCH_PADDING,rect[1],rect[2]+2*MATCH_PADDING,rect[3])

    img = utils.crop(src, rect)

    scores = []
    for hero in templates:
        res = cv2.matchTemplate(img, templates[hero], cv2.TM_CCOEFF_NORMED)
        scores.append((hero, np.max(res)))

    scores.sort(reverse=True, key=lambda s:s[1])
    # print(scores)
    score = scores[0]
    if score[1] > HERO_THRESHOLD:
        return score[0]
    else:
        return None

def read_heroes(img, rects, templates):
    heroes = []
    for player in rects:
        heroes.append(read_hero(img, rects[player], templates))

    return heroes

save_templates()
rects = read_rects()
templates = read_templates()

def process_hero(src):
    hero = read_hero(src, rects[1], templates)

    img = utils.crop(src, rects[1])

    return '{}'.format(
        utils.val_to_string(hero),
    ), img

def process_heroes(src):
    heroes = read_heroes(src, rects, templates)

    img = cv2.hconcat([
        utils.crop(src, HERO_RECT_LEFT),
        utils.crop(src, HERO_RECT_RIGHT),
    ], 2)

    heroes = ['{:>9}'.format(utils.val_to_string(h)) for h in heroes]

    return ' '.join(heroes), img

def save(start, end, code):
    hero = {'heroes': HEROES}
    for player in range(1,13):
        hero[player] = []

    for src, frame in utils.read_frames(start, end, code):
        heroes = read_heroes(src, rects, templates)

        for i, h in enumerate(heroes):
            hero[i+1].append(HEROES.index(h) if h is not None else -1)

        print('Frame {:d} analyzed'.format(frame))

    utils.save_data('hero', hero, start, end, code)

def refine(code):
    obj = utils.load_data('obj_r',0,None,code)
    hero = utils.load_data('hero',0,None,code)

    utils.extend_none(obj['status'], [hero[str(p)] for p in range(1,13)], size=0)

    for player in range(1,13):
        player = str(player)
        # Remove sudden changes, looser threshold because values are smaller
        hero[player] = utils.remove_outlier(hero[player], size=3, threshold=0.5, interp=False)

    utils.fix_disconnect(code, hero, -1)

    # remove heroA -1 heroA (resurrected/hacked/discorded/dead...)
    # remove heroA -1 heroB (hero change...)
    EFFECT_SIZE = 10 # NOTE: may need to tune
    for player in range(1,13):
        hs = np.array(hero[str(player)]) # Convert to np array
        for i in range(len(hs)-EFFECT_SIZE-1):
            if (
                hs[i] is not None and
                hs[i] > -1 and
                hs[i+EFFECT_SIZE+1] is not None and
                hs[i+EFFECT_SIZE+1] > -1 and
                np.any(hs[i+1:i+EFFECT_SIZE+1] == -1)
            ):
                hs[i+1:i+EFFECT_SIZE+1] = hs[i]

        hero[str(player)] = list(hs) # Convert back to list

    hero_src = utils.load_data('hero',0,None,code)
    plt.figure('status')
    plt.plot(obj['status'])
    plt.figure('hero team 1')
    for player in range(1,7):
        plt.subplot(6,1,player)
        plt.plot(hero[str(player)])
        plt.plot(hero_src[str(player)], '.', markersize=1)
    plt.figure('hero team 2')
    for player in range(7,13):
        plt.subplot(6,1,player-6)
        plt.plot(hero[str(player)])
        plt.plot(hero_src[str(player)], '.', markersize=1)
    plt.show()

    utils.save_data('hero_r', hero, 0, None, code)

# utils.read_batch(process_hero, start=0, map='volskaya', length=731)
# utils.read_batch(process_heroes, start=8, map='volskaya', length=731, num_width=2, num_height=32)
# save(0, None, 'volskaya')
# refine('volskaya')

# save(0, None, 'nepal')
# refine('nepal')

# utils.read_batch(process_heroes, start=1, map='hanamura', length=1623, num_width=2, num_height=32)
# save(0, None, 'hanamura')
# refine('hanamura')
