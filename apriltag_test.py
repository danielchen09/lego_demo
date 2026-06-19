from dt_apriltags import Detector
import numpy
import os
import cv2
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
    for tag in tags:
        print(tag)
        for idx in range(4):
            cv2.line(frame, tuple(tag.corners[idx - 1, :].astype(int)),
                     tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), 2)
        
        cv2.putText(frame, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int), tag.corners[0, 1].astype(int) - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0), thickness=2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows() 