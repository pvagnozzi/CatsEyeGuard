from typing import Sequence
from cv2.typing import MatLike

import cv2
import numpy as np

class MotionArea:
    _x1: int
    _y1: int
    _x2: int
    _y2: int

    def __init__(self, x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0):
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2

    @property
    def x1(self) -> int:
        return self._x1

    @property
    def y1(self) -> int:
        return self._y1

    @property
    def x2(self) -> int:
        return self._x2

    @property
    def y2(self) -> int:
        return self._y2

    def merge(self, other: 'MotionArea') -> 'MotionArea':
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)
        return MotionArea(x1, y1, x2, y2)

    def is_none(self):
        return self.x1 == 0 and self.y1 == 0 and self.x2 == 0 and self.y2 == 0

class MotionDetector:
    _old_image: MatLike
    _dilatation_kernel: MatLike

    _blur_size: int
    _min_size: int
    _threshold: float

    def __init__(self, min_size: int = 1000, threshold: int = 20, blur_size: int = 5) -> None:
        self._dilatation_kernel = np.ones((5, 5), np.uint8)
        self._blur_size = blur_size
        self._min_size = min_size
        self._threshold = threshold
        self._old_image = None

    # Detects the areas with motion in the image
    def detect_areas(self, image: MatLike) -> Sequence[MotionArea]:
        # Converts to gray
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self._old_image is None:
            self._old_image = gray_image
            return []

        # Image difference
        diff = cv2.absdiff(gray_image, self._old_image)

        # Save the current image for the next iteration
        self._old_image = gray_image

        # Gaussian blur for noise reduction
        blur = cv2.GaussianBlur(diff, (self._blur_size, self._blur_size), 0)

        # Apply a threshold for get a binary image
        _, thresh = cv2.threshold(blur, self._threshold, 255, cv2.THRESH_BINARY)

        # Image dilatation for filling holes
        dilated = cv2.dilate(thresh, self._dilatation_kernel, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < self._min_size:
                continue

            rect = cv2.boundingRect(contour)
            x1 = rect[0]
            y1 = rect[1]
            x2 = x1 + rect[2]
            y2 = y1 + rect[3]
            yield MotionArea(x1, y1, x2, y2)

    # Detects the area with motion in the image
    def detect_area(self, image: MatLike) -> MotionArea | None:
        result = MotionArea()

        rects = self.detect_areas(image)
        for rect in rects:
            if not result.is_none():
                result = result.merge(rect)
            else:
                result = rect
            result = result.merge(rect)

        return result if result is not None and not result.is_none() else None

    # Detects the area with motion in the image and crops it
    def detect_image(self, image: MatLike) -> MatLike | None:
        area = self.detect_area(image)
        return image[area.y1:area.y2, area.x1:area.x2] if area is not None else None
