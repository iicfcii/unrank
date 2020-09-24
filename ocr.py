import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

HEALTH_RECT = (170, 605, 100, 25)
ALPHA = 1.5 # Scale
BETA = -127*ALPHA+50 # Offset


def crop(img, rect):
    img_crop = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    return img_crop

img = cv2.imread('img/overwatch_1_1_'+str(12000)+'.jpg', cv2.IMREAD_GRAYSCALE)
img = crop(img, HEALTH_RECT)
img = cv2.convertScaleAbs(img, alpha=ALPHA, beta=BETA)
ret, img = cv2.threshold(img,160,255,cv2.THRESH_BINARY_INV)
img = cv2.resize(img, None, fx=6, fy=6)
cv2.imshow('img', img)
cv2.waitKey(0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(pytesseract.image_to_string(img))
