# 260201059 Ertuğrul Demir
import numpy as np
import cv2 as cv


def warpImages(img1, img2, H):  # From https://datahacker.rs/005-how-to-create-a-panorama-image-using-opencv-with-python/
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    list_of_points_2 = cv.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

    output_img = cv.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

    return output_img

def exercise_3():
    sift = cv.SIFT_create()

    center = cv.imread('../data/goldengate-03.png') # center is 2 result is very bad if center is 3 result better than second one therefore center is 3.
    center = cv.cvtColor(center, cv.COLOR_BGR2GRAY)
    for i in range(6):
        if i == 3: continue
        keypoints_2, descriptors_2 = sift.detectAndCompute(center, None)

        img1 = cv.imread('../data/goldengate-0'+str(i)+'.png')
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
        bf = cv.BFMatcher()
        matches = bf.match(descriptors_1,descriptors_2)

        src_pts = np.float32([ keypoints_1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints_2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)

        center = warpImages(center,img1,M)

    cv.imwrite('../results/panaroma.png',center)


exercise_3()
