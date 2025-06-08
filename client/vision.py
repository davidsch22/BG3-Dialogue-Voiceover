import cv2 as cv
import numpy as np
from filters import HsvFilter


class Vision:
    # Constants
    TRACKBAR_WINDOW = "Trackbars"

    # Constructor
    def __init__(self):
        # Initialize the trackbar window
        # self.init_control_gui()
        pass

    # Create GUI window with controls for adjusting arguments in real-time
    def init_control_gui(self):
        cv.namedWindow(self.TRACKBAR_WINDOW, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.TRACKBAR_WINDOW, 350, 700)

        # Required callback. we'll be using getTrackbarPos() to do lookups
        # instead of using the callback.
        def nothing(position):
            pass

        # Create trackbars for bracketing.
        # OpenCV scale for HSV is H: 0-179, S: 0-255, V: 0-255
        cv.createTrackbar('HMin', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('HMax', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMax', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMax', self.TRACKBAR_WINDOW, 0, 255, nothing)

        # Set default value for Max HSV trackbars
        cv.setTrackbarPos('HMax', self.TRACKBAR_WINDOW, 179)
        cv.setTrackbarPos('SMax', self.TRACKBAR_WINDOW, 255)
        cv.setTrackbarPos('VMax', self.TRACKBAR_WINDOW, 255)

        # Trackbars for increasing/decreasing saturation and value
        cv.createTrackbar('SAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('SSub', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VSub', self.TRACKBAR_WINDOW, 0, 255, nothing)

        cv.createTrackbar('Thresh 1', self.TRACKBAR_WINDOW, 0, 2000, nothing)
        cv.createTrackbar('Thresh 2', self.TRACKBAR_WINDOW, 0, 2000, nothing)
        cv.setTrackbarPos('Thresh 2', self.TRACKBAR_WINDOW, 2000)

    # Returns an HSV filter object based on the control GUI values
    def get_hsv_filter_from_controls(self):
        # Get current positions of all trackbars
        hsv_filter = HsvFilter()
        hsv_filter.hMin = cv.getTrackbarPos('HMin', self.TRACKBAR_WINDOW)
        hsv_filter.sMin = cv.getTrackbarPos('SMin', self.TRACKBAR_WINDOW)
        hsv_filter.vMin = cv.getTrackbarPos('VMin', self.TRACKBAR_WINDOW)
        hsv_filter.hMax = cv.getTrackbarPos('HMax', self.TRACKBAR_WINDOW)
        hsv_filter.sMax = cv.getTrackbarPos('SMax', self.TRACKBAR_WINDOW)
        hsv_filter.vMax = cv.getTrackbarPos('VMax', self.TRACKBAR_WINDOW)
        hsv_filter.sAdd = cv.getTrackbarPos('SAdd', self.TRACKBAR_WINDOW)
        hsv_filter.sSub = cv.getTrackbarPos('SSub', self.TRACKBAR_WINDOW)
        hsv_filter.vAdd = cv.getTrackbarPos('VAdd', self.TRACKBAR_WINDOW)
        hsv_filter.vSub = cv.getTrackbarPos('VSub', self.TRACKBAR_WINDOW)
        return hsv_filter

    # Given an image and an HSV filter, apply the filter and return the resulting image.
    # If a filter is not supplied, the control GUI trackbars will be used
    def apply_hsv_filter(self, input_img, hsv_filter=None):
        # Convert image to HSV
        hsv = cv.cvtColor(input_img, cv.COLOR_BGR2HSV)

        # If we haven't been given a defined filter, use the filter values from the GUI
        if not hsv_filter:
            hsv_filter = self.get_hsv_filter_from_controls()

        # Add/subtract saturation and value
        h, s, v = cv.split(hsv)
        s = self.shift_channel(s, hsv_filter.sAdd)
        s = self.shift_channel(s, -hsv_filter.sSub)
        v = self.shift_channel(v, hsv_filter.vAdd)
        v = self.shift_channel(v, -hsv_filter.vSub)
        hsv = cv.merge([h, s, v])

        # Set minimum and maximum HSV values to display
        lower = np.array([hsv_filter.hMin, hsv_filter.sMin, hsv_filter.vMin])
        upper = np.array([hsv_filter.hMax, hsv_filter.sMax, hsv_filter.vMax])

        # Apply the thresholds
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(hsv, hsv, mask=mask)

        # Convert to BGR for imshow() to display it properly
        img = cv.cvtColor(result, cv.COLOR_HSV2BGR)

        return img

    # Apply adjustments to an HSV channel
    # https://stackoverflow.com/questions/49697363/shifting-hsv-pixel-values-in-python-using-numpy
    def shift_channel(self, c, amount):
        if amount > 0:
            lim = 255 - amount
            c[c >= lim] = 255
            c[c < lim] += amount
        elif amount < 0:
            amount = -amount
            lim = amount
            c[c <= lim] = 0
            c[c > lim] -= amount
        return c

    def crop_text(self, input_img: np.ndarray, hsv_filter=None):
        result = cv.bilateralFilter(input_img, 3, 500, 500)

        result = cv.cvtColor(result, cv.COLOR_BGR2HSV)

        if not hsv_filter:
            hsv_filter = self.get_hsv_filter_from_controls()

        # Set minimum and maximum HSV values to display
        text_lower = np.array(
            [hsv_filter.hMin, hsv_filter.sMin, hsv_filter.vMin])
        text_upper = np.array(
            [hsv_filter.hMax, hsv_filter.sMax, hsv_filter.vMax])

        new_color = np.array([0, 0, 0])

        # Apply the thresholds
        text_mask = cv.inRange(result, text_lower, text_upper)
        mask_inv = cv.bitwise_not(text_mask)
        result[mask_inv > 0] = new_color

        result = cv.cvtColor(result, cv.COLOR_HSV2BGR)

        result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
        thresh1 = 0
        thresh2 = 497
        # thresh1 = cv.getTrackbarPos('Thresh 1', self.TRACKBAR_WINDOW)
        # thresh2 = cv.getTrackbarPos('Thresh 2', self.TRACKBAR_WINDOW)
        result = cv.Canny(result, thresh1, thresh2, L2gradient=True)

        kernel = np.ones((3, 16), np.uint8)
        result = cv.dilate(result, kernel)

        # Finding contours
        contours, hierarchy = cv.findContours(
            result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Get the largest contour cause it's likely the dialogue text
        if len(contours) == 0:
            return None
        text_contour = max(contours, key=cv.contourArea)

        if text_contour is None:
            return None

        # Return the original image cropped by the contour
        text_img = input_img.copy()
        mask = np.zeros_like(input_img)
        cv.drawContours(mask, [text_contour],
                        0, (255, 255, 255), -1)

        text_img = cv.bitwise_and(input_img, mask)

        (x, y, w, h) = cv.boundingRect(text_contour)
        text_img = text_img[y:y+h, x:x+w]
        return text_img

    def filter_text(self, input_img: np.ndarray, hsv_filter=None):
        text_img = cv.bilateralFilter(input_img, 3, 50, 50)

        if not hsv_filter:
            hsv_filter = self.get_hsv_filter_from_controls()

        text_img = self.apply_hsv_filter(
            text_img, hsv_filter)

        text_img = cv.cvtColor(text_img, cv.COLOR_BGR2GRAY)

        text_img = cv.resize(text_img, (0, 0), fx=1.5, fy=1.5)

        text_img = cv.copyMakeBorder(
            text_img, 5, 5, 5, 5, cv.BORDER_CONSTANT, None, (0, 0, 0))

        erode_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        text_img = cv.erode(text_img, erode_kernel)

        text_img = cv.bitwise_not(text_img)

        return text_img
