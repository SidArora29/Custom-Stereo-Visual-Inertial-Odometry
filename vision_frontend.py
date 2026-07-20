import os
import cv2
import numpy as np

left_path = '/media/tonyox/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam0/data/'

image_files = sorted([f for f in os.listdir(left_path) if f.endswith('.png')])

first_image_path = os.path.join(left_path, image_files[0])
old_frame = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)

p0 = cv2.goodFeaturesToTrack(old_frame, maxCorners=150, qualityLevel=0.3, minDistance=7)

mask = np.zeros_like(old_frame)

for i in range(1, len(image_files)):
    new_image_path = os.path.join(left_path, image_files[i])
    new_frame = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, p0, None)

    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 255, 2)
            new_frame = cv2.circle(new_frame, (int(a), int(b)), 5, 255, -1)

        img = cv2.add(new_frame, mask)
        cv2.imshow('Optical Flow Tracker', img)
        
        if cv2.waitKey(30) & 0xff == 27:
            break

        old_frame = new_frame.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if len(p0) < 100:
            p0 = cv2.goodFeaturesToTrack(new_frame, maxCorners=150, qualityLevel=0.3, minDistance=7)

cv2.destroyAllWindows()