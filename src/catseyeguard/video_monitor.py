import time
import cv2
import motion_detector as md


def video_monitor():
    cap = cv2.VideoCapture(0)
    motion_detector = md.MotionDetector()

    while True:
        ret, frame = cap.read()
        img = motion_detector.detect_image(frame)

        if img is not None:
            cv2.imshow('frame', img)
            time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()