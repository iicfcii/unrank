import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

from utils import crop, match_color

PROGRESS_RECT_1 = (592,71,46,46)
PROGRESS_RECT_2 = (642,71,46,46)

ICON_CENTER = (int(PROGRESS_RECT_1[2]/2), int(PROGRESS_RECT_1[3]/2))
ICON_RECT_1 = (601,81,26,26)
ICON_RECT_2 = (651,81,26,26)
ICON_RECT_OFFSET = (ICON_RECT_1[0]-PROGRESS_RECT_1[0], ICON_RECT_1[1]-PROGRESS_RECT_1[1])
ICON_MASK = np.array([[ICON_RECT_1[2]/2-1, 2],
                      [ICON_RECT_1[2]-3, ICON_RECT_1[3]/2],
                      [ICON_RECT_1[2]/2-1, ICON_RECT_1[3]-2],
                      [1, ICON_RECT_1[3]/2]], dtype=np.int32)
ICON_THRESHOLD = 0.4 # NOTE: A,B icon change size a bit when point is contesting


TEAM1_COLOR = np.array((85.0, 150.0, 255)) # HSV, RGB 67, 212, 255
TEAM1_COLOR_RANGE = np.array((20.0, 120, 40))
TEAM1_COLOR_LB = TEAM1_COLOR-TEAM1_COLOR_RANGE
TEAM1_COLOR_UB = TEAM1_COLOR+TEAM1_COLOR_RANGE

TEAM2_COLOR = np.array((170.0, 180, 230)) # HSV, RGB 240, 14, 54
TEAM2_COLOR_RANGE = np.array((20.0, 120, 40))
TEAM2_COLOR_LB = TEAM2_COLOR-TEAM2_COLOR_RANGE
TEAM2_COLOR_UB = TEAM2_COLOR+TEAM2_COLOR_RANGE

def save_assult_templates():
    img = cv2.imread('img/hanamura/hanamura_660.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_locked.jpg', crop(img, ICON_RECT_1))

    img = cv2.imread('img/hanamura/hanamura_4440.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_captured.jpg', crop(img, ICON_RECT_1))

    img = cv2.imread('img/hanamura/hanamura_3240.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_A1.jpg', crop(img, ICON_RECT_1))

    img = cv2.imread('img/hanamura/hanamura_12060.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_A2.jpg', crop(img, ICON_RECT_1))

    img = cv2.imread('img/hanamura/hanamura_4440.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_B1.jpg', crop(img, ICON_RECT_2))

    img = cv2.imread('img/hanamura/hanamura_18240.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_B2.jpg', crop(img, ICON_RECT_2))

def read_assult_tempaltes():
    templates = {}

    img_locked = cv2.imread('template/assult_locked.jpg', cv2.IMREAD_COLOR)
    mask_locked = np.zeros((img_locked.shape[0],img_locked.shape[1]), dtype=np.uint8)
    mask_locked = cv2.fillConvexPoly(mask_locked, ICON_MASK, 255)
    templates['locked'] = (img_locked, mask_locked)

    img_captured = cv2.imread('template/assult_captured.jpg', cv2.IMREAD_COLOR)
    mask_captured= np.zeros((img_captured.shape[0],img_captured.shape[1]), dtype=np.uint8)
    mask_captured = cv2.fillConvexPoly(mask_captured, ICON_MASK, 255)
    templates['captured'] = (img_captured, mask_captured)

    img_A1 = cv2.imread('template/assult_A1.jpg', cv2.IMREAD_COLOR)
    mask_A1= np.zeros((img_A1.shape[0],img_A1.shape[1]), dtype=np.uint8)
    mask_A1 = cv2.fillConvexPoly(mask_A1, ICON_MASK, 255)
    templates['A1'] = (img_A1, mask_A1)

    img_A2 = cv2.imread('template/assult_A2.jpg', cv2.IMREAD_COLOR)
    mask_A2= np.zeros((img_A2.shape[0],img_A2.shape[1]), dtype=np.uint8)
    mask_A2 = cv2.fillConvexPoly(mask_A2, ICON_MASK, 255)
    templates['A2'] = (img_A2, mask_A2)

    img_B1 = cv2.imread('template/assult_B1.jpg', cv2.IMREAD_COLOR)
    mask_B1= np.zeros((img_B1.shape[0],img_B1.shape[1]), dtype=np.uint8)
    mask_B1 = cv2.fillConvexPoly(mask_B1, ICON_MASK, 255)
    templates['B1'] = (img_B1, mask_B1)

    img_B2 = cv2.imread('template/assult_B2.jpg', cv2.IMREAD_COLOR)
    mask_B2= np.zeros((img_B2.shape[0],img_B2.shape[1]), dtype=np.uint8)
    mask_B2 = cv2.fillConvexPoly(mask_B2, ICON_MASK, 255)
    templates['B2'] = (img_B2, mask_B2)

    # img_locked[mask_locked == 0] = (0,0,0)
    # img_captured[mask_captured == 0] = (0,0,0)
    # img_A1[mask_A1 == 0] = (0,0,0)
    # img_A2[mask_A2 == 0] = (0,0,0)
    # img_B1[mask_B1 == 0] = (0,0,0)
    # img_B2[mask_B2 == 0] = (0,0,0)
    # cv2.imshow('locked with mask', img_locked)
    # cv2.imshow('captured with mask', img_captured)
    # cv2.imshow('A1 with mask', img_A1)
    # cv2.imshow('A2 with mask', img_A2)
    # cv2.imshow('B1 with mask', img_B1)
    # cv2.imshow('B2 with mask', img_B2)
    # cv2.waitKey(0)

    return templates

def read_point_status(img, templates, point):
    if point == 'A':
        img = crop(img, PROGRESS_RECT_1)
    else:
        img = crop(img, PROGRESS_RECT_2)

    scores = []
    template, mask = templates['locked']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    loc = np.unravel_index(np.argmax(res), res.shape)
    scores.append((-1, None, loc, res[loc]))

    template, mask = templates['captured']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    loc = np.unravel_index(np.argmax(res), res.shape)
    scores.append((100, None, loc, res[loc]))

    template, mask = templates[point+'1']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    loc = np.unravel_index(np.argmax(res), res.shape)
    scores.append((0, 2, loc, res[loc])) # attacker team is 2

    template, mask = templates[point+'2']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    loc = np.unravel_index(np.argmax(res), res.shape)
    scores.append((0, 1, loc, res[loc])) # attacker team is 1

    scores.sort(reverse=True, key=lambda m:m[3])

    if scores[0][3] > ICON_THRESHOLD:
        return scores[0][0],scores[0][1],(scores[0][2][1],scores[0][2][0])
    else:
        return None, None, None

def read_progress(img, templates):
    progress = {'A': -1, 'B': -1, 'attacker': None}
    # Match point status(locked, captured)
    progress['A'], attacking_A, loc_A = read_point_status(img, templates, 'A')
    progress['B'], attacking_B, loc_B = read_point_status(img, templates, 'B')

    if progress['A'] is not None and progress['A'] == 0:
        progress['attacker'] = attacking_A
    if progress['B'] is not None and progress['B'] == 0:
        progress['attacker'] = attacking_B

    imgs = {'A':crop(img, PROGRESS_RECT_1), 'B':crop(img, PROGRESS_RECT_2)}
    img_both = cv2.hconcat([imgs['A'], imgs['B']], 2)

    if progress['A'] is None or progress['B'] is None:
        return progress, img_both

    # Check which point to read
    if progress['A'] == 0 and progress['B'] == -1:
        point_current = 'A'
        loc = loc_A
    elif progress['A'] == 100 and progress['B'] == 0:
        point_current = 'B'
        loc = loc_B
    else:
        return progress, img_both
    center = (ICON_CENTER[0], ICON_CENTER[1]+loc[1]-ICON_RECT_OFFSET[1]) # Overtime will offset in y direction
    img_current = imgs[point_current].copy()
    mask= np.zeros((img_current.shape[0],img_current.shape[1]), dtype=np.uint8)
    mask = cv2.circle(mask, center, 16, 255, thickness=4)
    img_current[mask == 0] = (0,0,0)

    if progress['attacker'] == 1:
        img_match = match_color(img_current, TEAM1_COLOR_LB, TEAM1_COLOR_UB)
    else:
        img_match = match_color(img_current, TEAM2_COLOR_LB, TEAM2_COLOR_UB)
    img_match = cv2.morphologyEx(img_match, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)), iterations=1, borderValue=0)

    # Convert all valid pixles to angle
    pts = np.transpose(np.array((img_match > 0).nonzero()))
    pts[:,[0,1]] = pts[:,[1,0]] # Swap so that pt=(x,y)
    # Calculate angle using atan2(y,x)
    # bottom to top is x, left to right is y
    theta = np.arctan2(pts[:,0]-center[0], -(pts[:,1]-center[1]))
    # Convert to 0-2pi
    theta[theta < 0] += 2*np.pi
    # Sort and covnert to degrees
    theta = np.sort(theta)/np.pi*180
    if theta.shape[0] > 1:
        # Use average delta theta and initial angle to filter miss match
        dtheta = theta[1:]-theta[0:-1]

        # Mean of dtheta indicates the scatterness of data
        if np.mean(dtheta) < 5 and theta[0] < 15:
            start = -1
            indices = (dtheta > 20).nonzero()[0] # If big gap, use value before gap
            if len(indices) > 0:
                start = indices[0]
            progress[point_current] = int(theta[start]/360*100)

    img_current[img_match > 0] = (255,0,0)
    img_none = np.zeros((img_current.shape[0],img_current.shape[1], 3),dtype=np.uint8)
    if point_current == 'A':
        return progress, cv2.hconcat([img_current, img_none], 2)
    else:
        return progress, cv2.hconcat([img_none, img_current], 2)

    return progress, img_both

def read_batch(num_width=8, num_height=16):
    templates = read_assult_tempaltes()

    for i in range(0,int(1623/num_width/num_height)+1):
        imgs = []
        for j in range(i*num_width*num_height, i*num_width*num_height+num_width*num_height):
            img = cv2.imread('img/hanamura/hanamura_'+str(j*30)+'.jpg', cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((PROGRESS_RECT_1[3]*2, PROGRESS_RECT_1[2]*2, 3), dtype=np.uint8)
                elim = 0
            else:
                print(j*30)
                progress, img = read_progress(img, templates)

            info = str(progress['A'])+' '+str(progress['B'])+' '+str(progress['attacker'])
            if j < 1623:
                img = cv2.putText(img, info, (0,img.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                img = cv2.putText(img, str(j*30), (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            imgs.append(img)

        imgs_row = []
        for i in range(num_height):
            imgs_row.append(cv2.hconcat(imgs[i*num_width:i*num_width+num_width], num_width))

        img_final = cv2.vconcat(imgs_row, num_height)
        cv2.imshow('read', img_final)
        cv2.waitKey(0)

save_assult_templates()
# read_assult_tempaltes()
read_batch()
