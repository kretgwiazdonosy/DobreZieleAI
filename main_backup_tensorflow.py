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

    def screenCap(self, bbox, scale):
        """
        Makes a screen capture
        :param bbox: 4 corners of the rectangle that will be used as a capture, preferably a tuple
        """

        screen = np.array(ImageGrab.grab(bbox=bbox))
        screen = cv2.cvtColor(src=screen, code=cv2.COLOR_BGR2RGB)
        screen = cv2.resize(screen, None, fx=scale, fy=scale)

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

    def findContours(self, src):
        """
        Retrieves contours
        :param src: Source img to be searched for contours (data type: CV_8UC1)
        :return:
        """

        contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def write(self, contours, mask, templ=None, dst=None):

        """
        Method used solely to modify the destination (with everything that has been computed here)
        :param contours: Located contours to put on the destination
        :param mask: Mask of the destination
        :param templ: Dictionary of templates that will be searched for
        :param dst: Destination to be modified
        :return:
        """

        for contour in contours:
            if cv2.contourArea(contour) < 80:
                continue

            #
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255))

            roi = mask[y:y+h, x:x+w]

            if templ:

                for template in templ.values():

                    location = GameScreen.locateTemplate(None, roi, template, (x, y))
                    dst = screen.writeTemplate(template, location, dst)
                    cv2.putText(dst, 'contour', (x + int(0.5 * w), y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (83, 23, 255))

            else:
                cv2.putText(dst, 'contour', (x + int(0.5*w), y-5), cv2.FONT_HERSHEY_PLAIN, 1, (83, 23, 255))

        return dst

    def locateTemplate(self, src, template, pos):
        """
        Locate position of given template on the source image. In this case, it'll be used to locate both leaves and
        platforms, respectively putting them in their own variables. Method returns only those regions where the chance
        that template is located exceeds 40%
        :param template: Path to img containing template that will be searched in given source
        :param src: Source which is being searched for template
        :param pos: Tuple containing position of ROI in original img. Two elements in tuple
        :return:
        """

        locations = list()
        average = [0, 0]

        template = cv2.imread(template)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        try:

            candidates = cv2.matchTemplate(src, template, cv2.TM_CCOEFF_NORMED)
            result = np.where(candidates >= 0.6)

            for coordinates in zip(*result):

                locations.append(coordinates)

            for i in range(len(locations)):
                average[0] += locations[i][0]
                average[1] += locations[i][1]

            try:

                average[0] //= len(locations)
                average[1] //= len(locations)

                # Adding to coordinates actual position of ROI. The indexes are inverted because average stores data
                # in (y, x) format while pos uses (x, y). Too easy to fix, too small to care
                average[0] += pos[1]
                average[1] += pos[0]

            except ZeroDivisionError: # Happens when no template was found in given ROI
                pass

        except cv2.error: # Happens when template is bigger than given ROI
            pass

        return average

    def writeTemplate(self, templ, obj, dst):
        """
        Method used to return actual point of object to final capture
        :param obj: Location of the object
        :param dst: Destination, capture which will be modified with :obj param: and returned.
        :return:
        """
        # true point of object, for now turned off
        print(f'obiekt: {obj}')
        dst[obj[0]-5:obj[0]+5, obj[1]-5:obj[1]+5] = 0

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

    cap = screen.screenCap(bbox=(3, 88, 811, 610), scale=0.6)
    mask = screen.maskCreate(src=cap)

    contours = screen.findContours(mask)
    cap = screen.write(contours, mask, templates, cap)

    cv2.imshow('game capture', cap)
    if cv2.waitKey(1) == ord('q'):
        break