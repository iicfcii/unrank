import cv2
import numpy as np
import matplotlib.pyplot as plt

import utils
from hero import HEROES

ELIM_RECT_1 = (46,92,18,10)
ELIM_RECT_7 = (860,92,18,10)
ELIM_RECT_X_OFFSET = 71

ELIMS_RECT = (910,110,350,210)

RATIO = 1.5 # Scale image to make threshold clearer
ELIM_THRESHOLD = 0.9
DIST_THRESHOLD = 30
SYMBOL_THRESHOLD = 0.38

# NOTE: tune tighter so that won't make mistake
# Elim will appear on multiple frames to make up the missed ones
TEAM_1_COLOR = np.array((100, 127, 127)) #HSV
TEAM_1_COLOR_RANGE = np.array((60, 128, 128))
TEAM_1_COLOR_LB = TEAM_1_COLOR-TEAM_1_COLOR_RANGE
TEAM_1_COLOR_UB = TEAM_1_COLOR+TEAM_1_COLOR_RANGE

TEAM_2_COLOR = np.array((170, 150, 160)) #HSV
TEAM_2_COLOR_RANGE = np.array((20, 100, 100))
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
    # cv2.imshow('Crop', utils.crop(img, rects[1]))
    # for i in range(1,13):
    #     img_mark = cv2.rectangle(img, rects[i], (255,255,255), thickness=1)
    # cv2.imshow('img', img_mark)
    # cv2.waitKey(0)

    return rects

def save_templates():
    rects = read_rects()

    img = cv2.imread('img/volskaya/volskaya_3060.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/elim_1.jpg', utils.crop(img, rects[3]))

    img = cv2.imread('img/volskaya/volskaya_15570.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/elim_7.jpg', utils.crop(img, rects[12]))

def read_templates():
    templates = {}

    templates['symbol'] = {}
    img = cv2.imread('template/elim_1.jpg', cv2.IMREAD_COLOR)
    img_match = utils.match_color(img, ELIM_COLOR_LB, ELIM_COLOR_UB)
    img[img_match == 0] = (127,127,127) # Set to middle so that any color won't differ too much
    templates['symbol'][1] = img

    img = cv2.imread('template/elim_7.jpg', cv2.IMREAD_COLOR)
    img_match = utils.match_color(img, ELIM_COLOR_LB, ELIM_COLOR_UB)
    img[img_match == 0] = (127,127,127)
    templates['symbol'][7] = img

    ratio = 0.7
    for hero in HEROES:
        img = cv2.imread('template/hero_'+hero+'.jpg', cv2.IMREAD_COLOR)
        # Scale the hero template to match the size of elim icon
        img = cv2.resize(img, None, fx=ratio, fy=ratio)
        # Scale for better matching
        img = cv2.resize(img, None, fx=RATIO, fy=RATIO)
        templates[hero] = img

    # cv2.imshow('elim', templates['symbol'][7])
    # cv2.imshow('hero', templates['doomfist'])
    # cv2.waitKey(0)

    return templates

def read_status(src, rects, templates):
    status = []

    for p in rects:
        img = utils.crop(src, utils.pad_rect(rects[p],2,0))
        res = cv2.matchTemplate(img, templates['symbol'][1 if p < 7 else 7], cv2.TM_CCOEFF_NORMED)
        if np.max(res) > SYMBOL_THRESHOLD:
            status.append(1)
        else:
            status.append(0)

    return status

def determine_team(img_elim):
    img_src = img_elim.copy()
    # Determine team by making sure image include vertical border and match team colors
    b = 2 # border thickness
    img_elim[b:img_elim.shape[0]-b,:] = (0,0,0)

    img_1 = utils.match_color(img_elim, TEAM_1_COLOR_LB, TEAM_1_COLOR_UB)
    img_2 = utils.match_color(img_elim, TEAM_2_COLOR_LB, TEAM_2_COLOR_UB)
    sum_1 = np.sum(img_1)/255
    sum_2 = np.sum(img_2)/255

    if np.abs(sum_1-sum_2) < 10:
        team = None # Diff between red and blue are too small
    else:
        team = int(np.argmax([sum_1,sum_2])+1)

    # print(sum_1, sum_2, team)
    # cv2.imshow('Src', img_src)
    # cv2.imshow('Border', img_elim)
    # cv2.imshow('Team 1', img_1)
    # cv2.imshow('Team 2', img_2)
    # cv2.waitKey(0)

    return team

def read_elims(src, rects, templates):
    img = utils.crop(src, ELIMS_RECT)
    status = read_status(src, rects, templates)
    if np.sum(status) == 0: return status, []

    img_scaled = cv2.resize(img, None, fx=RATIO, fy=RATIO)

    heroes = []
    for hero in HEROES:
        res = cv2.matchTemplate(img_scaled, templates[hero], cv2.TM_CCOEFF_NORMED)
        locations = utils.extract_locs(res, ELIM_THRESHOLD, DIST_THRESHOLD)

        for loc in locations:
            # Crop unscaled image
            rect_elim = np.array((
                loc[0]/RATIO,
                loc[1]/RATIO-6, # Offset to top of elim rect
                templates[hero].shape[1]/RATIO, # Use template width
                28 # Height of elim rect
            ),dtype=np.int32)
            img_elim = utils.crop(img, rect_elim)
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

    elims = []
    for y in sorted(list(heroes_rows.keys()), key=lambda k:k):
        heroes_row = heroes_rows[y]
        if len(heroes_row) > 2:
            print('More than two heros found in this row')
            continue

        if len(heroes_row) == 2:
            heroes_row.sort(key=lambda h:h[2][0])
            if heroes_row[0][1] != None and heroes_row[1][1] != None: # Ignore hero without team
                if heroes_row[0][1] != heroes_row[1][1]: # Ignore mercy resurrect
                    elims.append([h[0:2] for h in heroes_row])

        if len(heroes_row) == 1:
            h = heroes_row[0]
            if h[1] is not None and h[2][0] > 350: # Suicide with known team
                elims.append([h[0:2]])
            else:
                print('Only one hero found in this row')

    return status, elims

save_templates()
templates = read_templates()
rects = read_rects()

def process_status(src):
    status = read_status(src, rects, templates)

    imgs = []
    for p in rects:
        imgs.append(utils.crop(src, utils.pad_rect(rects[p],1,10)))

    status = ['{:<3d}'.format(s) for s in status]

    return ''.join(status), cv2.hconcat(imgs, 12)

def process_elims(src):
    img = utils.crop(src, ELIMS_RECT)
    status, elims = read_elims(src, rects, templates)

    if len(elims) == 0: return 'NA', img

    elims_str = []
    for elim in elims:
        if len(elim) == 1:
            elims_str.append('{:.3}{:d}'.format(elim[0][0],elim[0][1]))
        else:
            elims_str.append('{:.3}{:d}-{:.3}{:d}'.format(elim[0][0],elim[0][1],elim[1][0],elim[1][1]))

    return ' '.join(elims_str), img

def save(start, end, code):
    health = {}
    elim = {
        'heroes': HEROES,
        'data': [],
    }

    for player in range(1,13):
        health[player] = []

    for src, frame in utils.read_frames(start, end, code):
        status, elims = read_elims(src, rects, templates)

        for i, value in enumerate(status):
            health[i+1].append(1 if value == 0 else 0)

        elims_frame = []
        if len(elims) > 0:
            for e in elims:
                if len(e) == 2:
                    elims_frame.append([
                        [HEROES.index(e[0][0]), e[0][1]],
                        [HEROES.index(e[1][0]), e[1][1]]
                    ])
                else:
                    elims_frame.append([
                        None,
                        [HEROES.index(e[0][0]), e[0][1]]
                    ])

        elim['data'].append(elims_frame)
        print('Frame {:d} analyzed'.format(frame))

    utils.save_data('health', health, start, end, code)
    utils.save_data('elim', elim, start, end, code)

def refine(code):
    obj = utils.load_data('obj_r',0,None,code)
    hero = utils.load_data('hero_r',0,None,code)
    health = utils.load_data('health',0,None,code)
    elim = utils.load_data('elim',0,None,code)

    # Clean health data
    utils.extend_none(obj['status'], [health[str(p)] for p in range(1,13)], size=0)
    for player in range(1,13):
        health[str(player)] = utils.remove_outlier(health[str(player)],size=1)
    utils.fix_disconnect(code, health, None)


    # Find all the deaths for each player
    death = {}
    for player in range(1,13):
        death[player] = []
        h = health[str(player)]
        start = None
        for i in range(1, len(h)):
            if h[i] == 0 and h[i-1] == 1:
                start = i

            if h[i] == 1 and h[i-1] == 0:
                death[player].append((start, i))
                start = None

            if start is not None and h[i] is None:
                death[player].append((start, i))
                start = None

    elim_r = {'heroes': HEROES}
    for p_self in range(1,13):
        elim_r[str(p_self)] = [None]*len(health[str(p_self)])
        for d in death[p_self]:
            start, end = d
            team_self = 1 if p_self < 7 else 2

            # NOTE: Echo ult kill shows echo but hero shows duplicated hero.
            # No effect here because echo ulting won't die??
            # Hero only changes after respawn
            hs_self = list(set(hero[str(p_self)][start:end]))
            if -1 in hs_self:
                hs_self.remove(-1)
                print('Not sure hero', p_self, start, end)
            assert len(hs_self) == 1
            h_self = hs_self[0] # current hero
            h_opp = -1
            p_opp = -1

            for es in elim['data'][start:end]:
                if len(es) == 0: continue

                for e in es:
                    if h_self == e[1][0] and team_self == e[1][1]:
                        if e[0] is not None:
                            # NOTE: All the elimination info read from images
                            # related to this death should all be caused by the
                            # same hero
                            if h_opp != -1: assert h_opp == e[0][0]
                            h_opp = e[0][0]
                        else:
                            # Suicide
                            h_opp = None
                            p_opp = None

                        # No duplicate elim possible for a signle frame
                        break

            if h_opp is not None:
                # Find the player
                team_opp = 1 if p_self > 6 else 2
                # Extend search range for echo(hero changes when ulting)
                if HEROES[h_opp] == 'echo': start -= 10
                for p in range((team_opp-1)*6+1,(team_opp-1)*6+7):
                    if h_opp in hero[str(p)][start:end]:
                        p_opp = p

            if p_opp == -1 or h_opp == -1:
                print('Death {} of player {:d} {} caused by opponent player {:d} {}'.format(d, p_self, HEROES[h_self], p_opp, HEROES[h_opp]))

            elim_r[str(p_self)][start] = p_opp

    elim = elim_r

    health_src = utils.load_data('health',0,None,code)
    elim_src = utils.load_data('elim',0,None,code)

    plt.figure('team 1')
    for player in range(1,7):
        plt.subplot(6,1,player)
        plt.plot(health[str(player)])
        plt.plot(health_src[str(player)],'.', markersize=1)
        plt.plot(elim[str(player)],'v')

    plt.figure('team 2')
    for player in range(7,13):
        plt.subplot(6,1,player-6)
        plt.plot(health[str(player)])
        plt.plot(health_src[str(player)],'.', markersize=1)
        plt.plot(elim[str(player)],'v')
    plt.show()

    utils.save_data('health_r', health, 0, None, code)
    utils.save_data('elim_r', elim, 0, None, code)

# utils.read_batch(process_status, start=0, map='volskaya', length=731, num_width=3, num_height=16)
# utils.read_batch(process_elims, start=20, map='volskaya', length=731, num_width=5, num_height=4)

# save(0,None,'volskaya')
# refine('volskaya')

# save(0,None,'hanamura')
# refine('hanamura')

# utils.read_batch(process_elims, start=49, code='junkertown', num_width=5, num_height=4)
# save(0,None,'junkertown')
# refine('junkertown')

# utils.read_batch(process_elims, start=28, code='blizzardworld', num_width=5, num_height=4)
# save(0, None, 'blizzardworld')
# refine('blizzardworld')
