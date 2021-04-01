import cv2

import assault
import control
import escort
import hybrid
import elim
import hero
import ult
import map
import utils

code = 'Y1DZV3'
for src, frame in utils.read_frames(808, None, code):
    info, img = hero.process_heroes(src)
    print(info)
    cv2.imshow('img', img)
    cv2.waitKey(0)
# Redo refine
# hero.refine(code)
