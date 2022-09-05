"""
Idk mainframe or sth
"""

import math
from PIL import ImageGrab
import numpy as np
import cv2

class GameScreen:
    """
    Class consisting of game-screen data and processing, such as screen capture, or mask creation and management
    """

    def __init__(self):
        pass

    def screenCap(self, bbox):

        screen = np.array(ImageGrab.grab(bbox=bbox))
        screen = cv2.cvtColor(src=screen, code=cv2.COLOR_BGR2RGB)

        return screen

    def maskCreate(self, src):
        """
        Returns game-screen mask
        """
        b, g, r = cv2.split(src)

        _, red_mask = cv2.threshold(r, 80, 255, cv2.THRESH_BINARY)
        s, blue_mask = cv2.threshold(b, 100, 255, cv2.THRESH_BINARY)
        g, gray_mask = cv2.threshold(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), 180, 100, cv2.THRESH_BINARY)

        leaves = red_mask + blue_mask
        _, final_mask = cv2.threshold(cv2.bitwise_not(gray_mask + leaves), 5, 255, cv2.THRESH_BINARY)

        return final_mask

    def findContours(self, src, dst):

        contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 80:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255))

            cv2.putText(dst, 'contour', (x + int(0.5*w), y-5), cv2.FONT_HERSHEY_PLAIN, 1, (83, 23, 255))

        return dst

    def locateTemplate(self, src, template):
        """
        Locate position of given template on the source image. In this case, it'll be used to locate both leaves and
        platforms, respectively putting them in their own variables. Method returns only those regions where the chance
        that template is located exceeds 70%
        :param template:
        :param src:
        :return:
        """

        locations = list()

        template = cv2.imread(template)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        candidates = cv2.matchTemplate(src, template, cv2.TM_CCOEFF_NORMED)
        result = np.where(candidates >= 0.4)

        for coordinates in zip(*result):

            locations.append(coordinates)

        return locations

    def writeTemplate(self, templ, obj, dst):

        templ = cv2.imread(templ)
        templ = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)

        shape = np.shape(templ)

        for coordinates in obj:

            try:
                dst[coordinates[0]+int(0.5*shape[0]), coordinates[1]+int(0.35*shape[0])] = 0
            except IndexError:
                pass

        return dst


# Captured img size (LEFT, TOP, RIGHT, BOT)
bbox = (0, 80, 811, 610)

templates = {
    'leaf': 'templates/leaf_template.png',
    'platform': 'templates/platform_template.png',
    'wrzatek': 'templates/wrzatek_template.png',
    'coffee': 'templates/coffee_template.png',
    'guarana': 'templates/guarana_template.png'
}

screen = GameScreen()

while True:

    cap = screen.screenCap(bbox=(3, 88, 811, 610))
    mask = screen.maskCreate(src=cap)

    cap = screen.findContours(mask, cap)

    for template in templates.values():

        obj = screen.locateTemplate(mask, template)
        cap = screen.writeTemplate(template, obj, cap)

    cv2.imshow('game capture', cap)
    if cv2.waitKey(1) == ord('q'):
        break