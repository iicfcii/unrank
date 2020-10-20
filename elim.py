import cv2
import numpy as np
import time

from utils import crop, match_color, read_batch, val_to_string, extract_locs, pad_rect
from hero import HEROES

ELIM_RECT_1 = (46,92,18,10)
ELIM_RECT_7 = (860,92,18,10)
ELIM_RECT_X_OFFSET = 71

ELIMS_RECT = (910,110,350,210)

RATIO = 1.5 # Scale image to make threshold clearer
ELIM_THRESHOLD = 0.9
DIST_THRESHOLD = 30

# NOTE: tune tighter so that won't make mistake
# Elim will appear on multiple frames to make up the missed ones
TEAM_1_COLOR = np.array((100, 150, 200)) #HSV
TEAM_1_COLOR_RANGE = np.array((20, 100, 60))
TEAM_1_COLOR_LB = TEAM_1_COLOR-TEAM_1_COLOR_RANGE
TEAM_1_COLOR_UB = TEAM_1_COLOR+TEAM_1_COLOR_RANGE

TEAM_2_COLOR = np.array((170, 150, 160)) #HSV
TEAM_2_COLOR_RANGE = np.array((20, 50, 50))
TEAM_2_COLOR_LB = TEAM_2_COLOR-TEAM_2_COLOR_RANGE
TEAM_2_COLOR_UB = TEAM_2_COLOR+TEAM_2_COLOR_RANGE

ELIM_COLOR = np.array((175, 216, 180)) #HSV
ELIM_COLOR_RANGE = np.array((20, 80, 80))
ELIM_COLOR_LB = ELIM_COLOR-ELIM_COLOR_RANGE
ELIM_COLOR_UB = ELIM_COLOR+ELIM_COLOR_RANGE

def read_rects():
    rects = {}

    for i in range(1,7):
        rects[i] = (
            ELIM_RECT_1[0]+(i-1)*ELIM_RECT_X_OFFSET,
            ELIM_RECT_1[1],
            ELIM_RECT_1[2],
            ELIM_RECT_1[3]
        )

    for i in range(7,13):
        rects[i] = (
            ELIM_RECT_7[0]+(i-7)*ELIM_RECT_X_OFFSET,
            ELIM_RECT_7[1],
            ELIM_RECT_7[2],
            ELIM_RECT_7[3]
        )

    # img = cv2.imread('img/volskaya/volskaya_17400.jpg', cv2.IMREAD_COLOR)
    # img = cv2.imread('img/volskaya/volskaya_3030.jpg', cv2.IMREAD_COLOR)
    # cv2.imshow('crop', crop(img, rects[1]))
    # for i in range(1,13):
    #     img_mark = cv2.rectangle(img, rects[i], (255,255,255), thickness=1)
    # cv2.imshow('img', img_mark)
    # cv2.waitKey(0)

    return rects

def read_templates():
    templates = {}

    ratio = 0.7
    for hero in HEROES:
        img = cv2.imread('template/hero_'+hero+'.jpg', cv2.IMREAD_COLOR)
        # Scale the hero template to match the size of elim icon
        img = cv2.resize(img, None, fx=ratio, fy=ratio)
        # Scale for better matching
        img = cv2.resize(img, None, fx=RATIO, fy=RATIO)
        templates[hero] = img

    # cv2.imshow('hero', templates['doomfist'])
    # cv2.waitKey(0)

    return templates

def read_status(src, rects, templates):
    # 0 indicates no elim
    # 1 indicates MAY have elims
    status = 0
    rs = []

    for p in rects:
        img = crop(src, rects[p])
        img_match = match_color(img, ELIM_COLOR_LB, ELIM_COLOR_UB)
        r = (np.sum(img_match)/255)/(img_match.shape[0]*img_match.shape[1])
        rs.append(r)

    # print(rs)

    rs.sort(reverse=True, key=lambda r:r)
    if rs[0] > 0.3: status = 1

    return status

def determine_team(img_elim):
    # Determine team by making sure image include vertical border and match team colors
    b = 4 # border thickness
    img_elim[b:img_elim.shape[0]-b,:] = (0,0,0)

    img_1 = match_color(img_elim, TEAM_1_COLOR_LB, TEAM_1_COLOR_UB)
    img_2 = match_color(img_elim, TEAM_2_COLOR_LB, TEAM_2_COLOR_UB)
    sum_1 = np.sum(img_1)/255
    sum_2 = np.sum(img_2)/255
    if np.abs(sum_1-sum_2) < 10:
        # print(sum_1, sum_2)
        # cv2.imshow('Src', img_elim)
        # cv2.imshow('Team 1', img_1)
        # cv2.imshow('Team 2', img_2)
        # cv2.waitKey(0)
        return None # Diff between red and blue are too small

    team = np.argmax([sum_1,sum_2])+1

    # print(sum_1, sum_2, team)
    # cv2.imshow('Src', img_elim)
    # cv2.imshow('Team 1', img_1)
    # cv2.imshow('Team 2', img_2)
    # cv2.waitKey(0)

    return team

def read_elims(src, rects, templates):
    img = crop(src, ELIMS_RECT)
    status = read_status(src, rects, templates)
    if status == 0: return None

    img_scaled = cv2.resize(img, None, fx=RATIO, fy=RATIO)

    heroes = []
    for hero in HEROES:
        res = cv2.matchTemplate(img_scaled, templates[hero], cv2.TM_CCOEFF_NORMED)
        locations = extract_locs(res, ELIM_THRESHOLD, DIST_THRESHOLD)

        for loc in locations:
            # Crop unscaled image
            rect_elim = np.array((
                loc[0]/RATIO,
                loc[1]/RATIO-7, # Offset to top of elim rect
                templates[hero].shape[1]/RATIO, # Use template width
                29 # Height of elim rect
            ),dtype=np.int32)
            img_elim = crop(img, rect_elim)
            team = determine_team(img_elim)
            heroes.append((hero, team, loc))
            if team is None: print('Not sure about', hero, 'team')

    heroes_rows = {}
    for hero in heroes:
        y = hero[2][1]
        added = False
        for y_row in heroes_rows:
            if np.abs(y-y_row) < 5:
                heroes_rows[y_row].append(hero)
                added = True
                break
        if not added:
            heroes_rows[y] = [hero]

    heroes_organized = []
    for y in sorted(list(heroes_rows.keys()), key=lambda k:k):
        heroes_row = heroes_rows[y]
        if len(heroes_row) > 2:
            print('More than two heros found in this row')
            continue

        if len(heroes_row) == 2:
            heroes_row.sort(key=lambda h:h[2][0])
            if heroes_row[0][1] != None and heroes_row[1][1] != None: # Ignore hero without team
                if heroes_row[0][1] != heroes_row[1][1]: # Ignore mercy resurrect
                    heroes_organized.append([h[0:2] for h in heroes_row])

        if len(heroes_row) == 1:
            h = heroes_row[0]
            if h[1] is not None and h[2][0] > 350: # Suicide with known team
                heroes_organized.append([h[0:2]])
            else:
                print('Only one hero found in this row')

    if len(heroes_organized) == 0: heroes_organized = None

    return heroes_organized

templates = read_templates()
rects = read_rects()

def process_status(src):
    status = read_status(src, rects, templates)

    imgs = []
    for p in rects:
        imgs.append(crop(src, pad_rect(rects[p],0,10)))

    return '{:d}'.format(status), cv2.hconcat(imgs, 12)

def process_elims(src):
    img = crop(src, ELIMS_RECT)
    heroes = read_elims(src, rects, templates)

    if heroes is None: return 'NA', img

    heroes_str = []
    for hs in heroes:
        if len(hs) == 1:
            heroes_str.append('{:.3}{:d}'.format(hs[0][0],hs[0][1]))
        else:
            heroes_str.append('{:.3}{:d}-{:.3}{:d}'.format(hs[0][0],hs[0][1],hs[1][0],hs[1][1]))

    return ' '.join(heroes_str), img

# read_batch(process_status, start=1, map='volskaya', length=731, num_width=3, num_height=16)
# read_batch(process_elims, start=0, map='volskaya', length=731, num_width=5, num_height=4)
read_batch(process_elims, start=0, map='volskaya', length=731, num_width=5, num_height=4)
# read_batch(process_elims, start=0, map='rialto', length=470, num_width=5, num_height=4)
