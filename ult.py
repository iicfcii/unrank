import cv2
import numpy as np
import matplotlib.pyplot as plt

import utils
import hero

RATIO = 2.0

ULT_THRESHOLD = 0.7
ULT_RECT_1 = (33,52,24,20)
ULT_RECT_7 = (828,52,24,20)
ULT_RECT_X_OFFSET = 71

ULT_0_1_RECT = (43,52,12,20)
ULT_1_1_RECT = (46,52,9,20)
ULT_7_1_RECT = (44,52,10,20)
ULT_0_2_RECT = (838,52,12,20)
ULT_1_2_RECT = (841,52,9,20)
ULT_7_2_RECT = (839,52,10,20)
ULT_TILT_X = 6
ULT_TILT_Y = 14
ULT_0_MASK = np.array([
    [ULT_TILT_X, 0],
    [ULT_0_2_RECT[2]-1, 0],
    [ULT_0_2_RECT[2]-1, ULT_0_2_RECT[3]-ULT_TILT_Y-1],
    [ULT_0_2_RECT[2]-ULT_TILT_X-1, ULT_0_2_RECT[3]-1],
    [0, ULT_0_2_RECT[3]-1],
    [0, ULT_TILT_Y],
], dtype=np.int32)
ULT_1_MASK = np.array([
    [ULT_TILT_X, 0],
    [ULT_1_2_RECT[2]-1, 0],
    [ULT_1_2_RECT[2]-1, ULT_1_2_RECT[3]-ULT_TILT_Y-1],
    [ULT_1_2_RECT[2]-ULT_TILT_X-1, ULT_1_2_RECT[3]-1],
    [0, ULT_1_2_RECT[3]-1],
    [0, ULT_TILT_Y],
], dtype=np.int32)
ULT_7_MASK = np.array([
    [ULT_TILT_X, 0],
    [ULT_7_2_RECT[2]-1, 0],
    [ULT_7_2_RECT[2]-1, ULT_7_2_RECT[3]-ULT_TILT_Y-1],
    [ULT_7_2_RECT[2]-ULT_TILT_X-1, ULT_7_2_RECT[3]-1],
    [0, ULT_7_2_RECT[3]-1],
    [0, ULT_TILT_Y],
], dtype=np.int32)

def save_templates():
    img = cv2.imread('template_src/volskaya_1590.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_0_2.jpg', utils.crop(img, ULT_0_2_RECT))
    img = cv2.imread('template_src/volskaya_1620.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_1_2.jpg', utils.crop(img, ULT_1_2_RECT))
    img = cv2.imread('template_src/volskaya_1800.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_2_2.jpg', utils.crop(img, ULT_0_2_RECT))
    img = cv2.imread('template_src/volskaya_1860.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_3_2.jpg', utils.crop(img, ULT_0_2_RECT))
    img = cv2.imread('template_src/volskaya_1950.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_4_2.jpg', utils.crop(img, ULT_0_2_RECT))
    img = cv2.imread('template_src/volskaya_2250.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_5_2.jpg', utils.crop(img, ULT_0_2_RECT))
    img = cv2.imread('template_src/volskaya_1980.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_6_2.jpg', utils.crop(img, ULT_0_2_RECT))
    img = cv2.imread('template_src/volskaya_2010.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_7_2.jpg', utils.crop(img, ULT_7_2_RECT))
    img = cv2.imread('template_src/volskaya_2070.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_8_2.jpg', utils.crop(img, ULT_0_2_RECT))
    img = cv2.imread('template_src/volskaya_3630.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_9_2.jpg', utils.crop(img, ULT_0_2_RECT))

    img = cv2.imread('template_src/volskaya_1620.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_0_1.jpg', utils.crop(img, ULT_0_1_RECT))
    img = cv2.imread('template_src/volskaya_1710.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_1_1.jpg', utils.crop(img, ULT_1_1_RECT))
    img = cv2.imread('template_src/volskaya_1740.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_2_1.jpg', utils.crop(img, ULT_0_1_RECT))
    img = cv2.imread('template_src/volskaya_1860.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_3_1.jpg', utils.crop(img, ULT_0_1_RECT))
    img = cv2.imread('template_src/volskaya_3810.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_4_1.jpg', utils.crop(img, ULT_0_1_RECT))
    img = cv2.imread('template_src/volskaya_2160.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_5_1.jpg', utils.crop(img, ULT_0_1_RECT))
    img = cv2.imread('template_src/volskaya_4560.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_6_1.jpg', utils.crop(img, ULT_0_1_RECT))
    img = cv2.imread('template_src/volskaya_4740.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_7_1.jpg', utils.crop(img, ULT_7_1_RECT))
    img = cv2.imread('template_src/volskaya_4770.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_8_1.jpg', utils.crop(img, ULT_0_1_RECT))
    img = cv2.imread('template_src/volskaya_4620.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_9_1.jpg', utils.crop(img, ULT_0_1_RECT))


    rects = read_rects()
    img = cv2.imread('template_src/junkertown_23730.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_dva_1.jpg', utils.crop(img, rects[6]))
    img = cv2.imread('template_src/junkertown_6210.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/ult_dva_2.jpg', utils.crop(img, utils.offset_rect(rects[8], 2, 0)))

    # cv2.imshow('img', utils.crop(img, utils.offset_rect(rects[8], 2, 0)))
    # cv2.waitKey(0)

def read_tempaltes():
    templates = {}

    mask_0 = np.zeros((ULT_0_2_RECT[3],ULT_0_2_RECT[2]), dtype=np.uint8)
    mask_0 = cv2.fillConvexPoly(mask_0, ULT_0_MASK, 255)
    mask_0 = cv2.resize(mask_0, None, fx=RATIO, fy=RATIO)


    mask_1 = np.zeros((ULT_1_2_RECT[3],ULT_1_2_RECT[2]), dtype=np.uint8)
    mask_1 = cv2.fillConvexPoly(mask_1, ULT_1_MASK, 255)
    mask_1 = cv2.resize(mask_1, None, fx=RATIO, fy=RATIO)

    mask_7 = np.zeros((ULT_7_2_RECT[3],ULT_7_2_RECT[2]), dtype=np.uint8)
    mask_7 = cv2.fillConvexPoly(mask_7, ULT_7_MASK, 255)
    mask_7 = cv2.resize(mask_7, None, fx=RATIO, fy=RATIO)

    for num in range(10):
        templates[num] = {}
        for team in [1,2]:
            img = cv2.imread('template/ult_'+str(num)+'_'+str(team)+'.jpg', cv2.IMREAD_COLOR)
            img = cv2.resize(img, None, fx=RATIO, fy=RATIO)

            mask = mask_0
            if num == 1: mask = mask_1
            if num == 7: mask = mask_7
            templates[num][team] = (img, mask)

    templates['dva'] = {}
    for team in [1,2]:
        img = cv2.imread('template/ult_dva_'+str(team)+'.jpg', cv2.IMREAD_COLOR)
        templates['dva'][team] = (img, None)

    # for num in range(10):
    #     for team in [1,2]:
    #         temp, mask = templates[num][team]
    #         temp[mask == 0] = (0,0,0)
    #         cv2.imshow(str(num)+str(team), temp)
    # cv2.waitKey(0)

    return templates

def read_rects():
    rects = {}

    for i in range(1,7):
        rects[i] = (
            ULT_RECT_1[0]+(i-1)*ULT_RECT_X_OFFSET,
            ULT_RECT_1[1],
            ULT_RECT_1[2],
            ULT_RECT_1[3]
        )

    for i in range(7,13):
        rects[i] = (
            ULT_RECT_7[0]+(i-7)*ULT_RECT_X_OFFSET,
            ULT_RECT_7[1],
            ULT_RECT_7[2],
            ULT_RECT_7[3]
        )

    # img = cv2.imread('img/volskaya/volskaya_2370.jpg', cv2.IMREAD_COLOR)
    # for i in range(1,13):
    #     img_mark = cv2.rectangle(img, rects[i], (255,255,255), thickness=1)
    # cv2.imshow('img', img_mark)
    # cv2.waitKey(0)

    return rects

def read_ult(src, rect, templates):
    team = 2 if rect[0] >= ULT_RECT_7[0] else 1
    padx = 2

    digit_1_scores = []
    for num in range(10):
        template, mask = templates[num][team]
        w_digit_1 = int(template.shape[1]/RATIO)
        # Offset to the left a bit because of digit 1 is not exactly at the right of the rect
        digit_1_rect = (rect[0]+rect[2]-w_digit_1-padx-3, rect[1], w_digit_1+padx*2, rect[3])
        img_digit_1 = utils.crop(src, digit_1_rect)
        img_digit_1 = cv2.resize(img_digit_1, None, fx=RATIO, fy=RATIO)
        res = cv2.matchTemplate(img_digit_1, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        digit_1_scores.append((num, np.max(res)))
    digit_1_scores.sort(reverse=True, key=lambda s:s[1])
    # print('1', digit_1_scores)
    # if rect[0] == rects[1][0]:
    #     cv2.imshow('img', img_digit_1)
    #     cv2.waitKey(0)
    digit_1 = digit_1_scores[0][0]
    if digit_1_scores[0][1] < ULT_THRESHOLD or np.isnan(digit_1_scores[0][1]):
        return None

    digit_2_scores = []
    for num in range(10):
        template, mask = templates[num][team]
        w_digit_2 = int(template.shape[1]/RATIO)
        w_digit_1 = int(templates[digit_1][team][0].shape[1]/RATIO)
        # Offset to the left is smaller because the digits are tilted(two rects have overlap)
        digit_2_rect = (rect[0]+rect[2]-w_digit_1-w_digit_2-2, rect[1], w_digit_2+padx*2, rect[3])
        img_digit_2 = utils.crop(src, digit_2_rect)
        img_digit_2 = cv2.resize(img_digit_2, None, fx=RATIO, fy=RATIO)
        res = cv2.matchTemplate(img_digit_2, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        digit_2_scores.append((num, np.max(res)))
    digit_2_scores.sort(reverse=True, key=lambda s:s[1])
    # print('2', digit_2_scores)
    # if rect[0] == rects[8][0]:
    #     cv2.imshow('img', img_digit_2)
    #     cv2.waitKey(0)
    digit_2 = digit_2_scores[0][0]
    if digit_2_scores[0][1] < ULT_THRESHOLD or np.isnan(digit_2_scores[0][1]):
        digit_2 = 0

    percent = digit_2*10+digit_1

    return percent

def read_ults(src, rects, templates):
    percents = []
    for player in rects:
        percents.append(read_ult(src, rects[player], templates))

    return percents

def read_dva_ult(src, rect, templates):
    team = 1 if rect[0] < ULT_RECT_7[0] else 2
    img = utils.crop(src, utils.pad_rect(rect, 5, 5))
    template, mask = templates['dva'][team]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    if np.max(res) > 0.7:
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        return True
    else:
        return False

save_templates()
rects = read_rects()
templates = read_tempaltes()

def process_ult(src):
    percent = read_ult(src, rects[1], templates)

    rect = list(rects[1])
    padding = 10
    rect[0] -= padding
    rect[1] -= padding
    rect[2] += padding*2
    rect[3] += padding*2
    img = utils.crop(src, rect)

    return '{}'.format(
        utils.val_to_string(percent),
    ), img

def process_ults(src):
    percents = read_ults(src, rects, templates)
    percents = ['{:<6}'.format(utils.val_to_string(p)) for p in percents]

    imgs = []
    for player in rects:
        rect = list(rects[player])
        padding = 8
        rect[0] -= padding
        rect[1] -= padding
        rect[2] += padding*2
        rect[3] += padding*2

        imgs.append(utils.crop(src, rect))
    img = cv2.hconcat(imgs, 12)

    return ''.join(percents), img

def save(start, end, code):
    ult = {}
    for player in range(1,13):
        ult[player] = []

    for src, frame in utils.read_frames(start, end, code):
        percents = read_ults(src, rects, templates)

        for i, percent in enumerate(percents):
            ult[i+1].append(percent)

    utils.save_data('ult', ult, start, end, code)

def refine(code):
    obj = utils.load_data('obj_r',0,None,code)
    ult = utils.load_data('ult',0,None,code)

    # Remove data when black screen(status is None)
    utils.extend_none(obj['status'], [ult[str(p)] for p in range(1,13)], size=0)

    for player in range(1,13):
        # ult can become 100(None) at i frame and then 0 at i+1 frame
        ult[str(player)] = utils.remove_outlier(ult[str(player)], threshold=0.1)

    utils.fix_disconnect(code, ult, 0)

    for player in range(1,13):
        player = str(player)
        for i in range(len(ult[player])):
            if (
                ult[player][i] is None and
                obj['status'][i] is not None and
                ult[player][i-1] != 0 and
                ult[player][i-1] != None# After ult won't gain ult within 1 second
            ):
                ult[player][i] = 100
        # Remove resurrect, up and almost returns to original percent
        ult[player] = utils.remove_outlier(
            ult[player],
            size=5,
            types=['up'],
            threshold=0.3,
            min=100,
            duration=4
        )

    ult_src = utils.load_data('ult',0,None,code)

    plt.figure('ult team 1')
    for player in range(1,7):
        plt.subplot(6,1,player)
        plt.plot(ult[str(player)])
        plt.plot(ult_src[str(player)], '.', markersize=1)
    utils.save_fig(utils.file_path('fig_ult_1',0,len(ult['1'])-1,code,ext='png'))

    plt.figure('ult team 2')
    for player in range(7,13):
        plt.subplot(6,1,player-6)
        plt.plot(ult[str(player)])
        plt.plot(ult_src[str(player)], '.', markersize=1)
    utils.save_fig(utils.file_path('fig_ult_2',0,len(ult['1'])-1,code,ext='png'))
    # plt.show()

    utils.save_data('ult_r', ult, 0, None, code)

def use(code):
    ult_src = utils.load_data('ult_r',0,None,code)
    hero_src = utils.load_data('hero_r',0,None,code)

    use = {}
    for player in range(1,13):
        player = str(player)
        ult_data = ult_src[player]
        hero_data = hero_src[player]
        use[player] = [None]*len(ult_data)
        i = 2
        while i < len(ult_data):
            if ult_data[i-2] is None or ult_data[i] is None:
                pass
            else:
                if ult_data[i] - ult_data[i-2] < -60: # Use -2 in case ult discharge happens over three frames
                    def no_switch():
                        # Note: hero can be unknown which will be refined to previous hero
                        # causing ult back to 0 but hero still the same

                        # Hero has to be the same before and after ult use
                        return hero_data[i] == hero_data[i-1] and hero_data[i] == hero_data[i+1]

                    if hero.HEROES[hero_data[i-2]] == 'echo': # Duplicate effect takes about one frame
                        # Check if echo switches hero or duplicated hero just for one frame
                        for src, frame in utils.read_frames(i, i+1, code):
                            duplicate_hero = hero.read_duplicate_hero(src, hero.HEROES[hero_data[i+1]], hero.read_rects()[int(player)], hero.read_templates())
                            print('Check echo ult', player, i, duplicate_hero)
                        if duplicate_hero: use[player][i] = 1
                    elif hero.HEROES[hero_data[i]] == 'dva':
                        if no_switch():
                            # Check if dva actually used bomb and not demech or mech
                            used_ult = False
                            for src, frame in utils.read_frames(i-5, i+1, code):
                                if read_dva_ult(src, rects[int(player)], templates):
                                    used_ult = True
                            print('Checked dva ult', player, i, used_ult)
                            if used_ult: use[player][i] = 1
                    else:
                        if no_switch(): use[player][i] = 1

                    i += 1 # Skip next frame to avoid double counting ult drop
            i += 1

    plt.figure('ult use team 1 ')
    for player in range(1,7):
        plt.subplot(6,1,player)
        player = str(player)
        plt.plot(ult_src[player])
        plt.plot(use[player],'v')
    utils.save_fig(utils.file_path('fig_ult_use_1',0,len(ult_src['1'])-1,code,ext='png'))

    plt.figure('ult use team 2 ')
    for player in range(7,13):
        plt.subplot(6,1,player-6)
        player = str(player)
        plt.plot(ult_src[player])
        plt.plot(use[player],'v')
    utils.save_fig(utils.file_path('fig_ult_use_2',0,len(ult_src['1'])-1,code,ext='png'))
    # plt.show()

    utils.save_data('ult_use', use, 0, None, code)
