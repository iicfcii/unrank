import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time

from utils import crop, match_color

PROGRESS_RECT_1 = (592,71,46,46)
PROGRESS_RECT_2 = (642,71,46,46)

ICON_CENTER = (int(PROGRESS_RECT_1[2]/2-1), int(PROGRESS_RECT_1[3]/2-1))
ICON_RECT_1 = (602,81,26,26)
ICON_RECT_2 = (652,81,26,26)
ICON_MASK = np.array([[ICON_RECT_1[2]/2-1, 2],
                      [ICON_RECT_1[2]-3, ICON_RECT_1[3]/2],
                      [ICON_RECT_1[2]/2-1, ICON_RECT_1[3]-2],
                      [1, ICON_RECT_1[3]/2]], dtype=np.int32)
ICON_THRESHOLD = 0.7


TEAM1_COLOR = np.array((90.0, 150.0, 255)) # HSV, RGB 67, 212, 255
TEAM1_COLOR_RANGE = np.array((15.0, 120, 40))
TEAM1_COLOR_LB = TEAM1_COLOR-TEAM1_COLOR_RANGE
TEAM1_COLOR_UB = TEAM1_COLOR+TEAM1_COLOR_RANGE

def save_assult_templates():
    img = cv2.imread('img/hanamura/hanamura_660.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_locked.jpg', crop(img, ICON_RECT_1))

    img = cv2.imread('img/hanamura/hanamura_4440.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_captured.jpg', crop(img, ICON_RECT_1))

    img = cv2.imread('img/hanamura/hanamura_3240.jpg', cv2.IMREAD_COLOR)
    cv2.imwrite('template/assult_A1.jpg', crop(img, ICON_RECT_1))

    img = cv2.imread('img/hanamura/hanamura_12570.jpg', cv2.IMREAD_COLOR)
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
    scores.append((-1, np.max(res)))

    template, mask = templates['captured']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    scores.append((100, np.max(res)))

    template, mask = templates[point+'1']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    scores.append((0, np.max(res)))

    template, mask = templates[point+'2']
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    scores.append((0, np.max(res)))

    scores.sort(reverse=True, key=lambda m:m[1])

    if scores[0][1] > ICON_THRESHOLD:
        return scores[0][0]
    else:
        return None

def read_progress(img, templates):
    progress = {'A': -1, 'B': -1}
    # Match point status(locked, captured)
    progress['A'] = read_point_status(img, templates, 'A')
    progress['B'] = read_point_status(img, templates, 'B')

    imgs = {
        'A':crop(img, PROGRESS_RECT_1),
        'B':crop(img, PROGRESS_RECT_2),
    }

    img_both = cv2.hconcat([imgs['A'], imgs['B']], 2)
    if progress['A'] is None or progress['B'] is None:
        return progress, img_both

    if progress['A'] > -1 and progress['B'] == -1:
        point_current = 'A'
    elif progress['A'] == 100 and progress['B'] > -1:
        point_current = 'B'
    else:
        return progress, img_both

    img_current = imgs[point_current].copy()
    mask= np.zeros((img_current.shape[0],img_current.shape[1]), dtype=np.uint8)
    mask = cv2.circle(mask, ICON_CENTER, 16, 255, thickness=4)
    img_current[mask == 0] = (0,0,0)
    img_current = match_color(img_current, TEAM1_COLOR_LB, TEAM1_COLOR_UB)
    # img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)), iterations=1, borderValue=0)

    # Convert all valid pixles to angle
    pts = np.transpose(np.array((img_current > 0).nonzero()))
    pts[:,[0,1]] = pts[:,[1,0]] # Swap so that pt=(x,y)
    # Calculate angle using atan2(y,x)
    # bottom to top is x, left to right is y
    theta = np.arctan2(pts[:,0]-ICON_CENTER[0], -(pts[:,1]-ICON_CENTER[1]))
    # Convert to 0-2pi
    theta[theta < 0] += 2*np.pi
    # Sort and covnert to degrees
    theta = np.sort(theta)/np.pi*180
    if theta.shape[0] > 1:
        # Use average delta theta and initial angle to filter miss match
        dtheta = theta[1:]-theta[0:-1]
        if np.mean(dtheta) < 5 and theta[0] < 15:
            progress[point_current] = int(theta[-1]/360*100)

    return progress, img_both

def read_batch(num_width=8, num_height=8):
    templates = read_assult_tempaltes()

    for i in range(9,int(1623/num_width/num_height)+1):
        imgs = []
        for j in range(i*num_width*num_height, i*num_width*num_height+num_width*num_height):
            img = cv2.imread('img/hanamura/hanamura_'+str(j*30)+'.jpg', cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((PROGRESS_RECT_1[3]*2, PROGRESS_RECT_1[2]*2, 3), dtype=np.uint8)
                elim = 0
            else:
                # print(j*30)
                progress, img = read_progress(img, templates)

            if j < 1623: img = cv2.putText(img, str(progress['A'])+' '+str(progress['B']), (0,img.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            # if j < 1623: img = cv2.putText(img, str(j*30), (0,img.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            imgs.append(img)

        imgs_row = []
        for i in range(num_height):
            imgs_row.append(cv2.hconcat(imgs[i*num_width:i*num_width+num_width], num_width))

        img_final = cv2.vconcat(imgs_row, num_height)
        cv2.imshow('read', img_final)
        cv2.waitKey(0)

# save_assult_templates()
# read_assult_tempaltes()
read_batch()
