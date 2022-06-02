# 260201059 Ertuğrul Demir
import cv2 as cv
import pickle as cp


def exercise_1():
    # Exercise 1
    sift = cv.SIFT_create()

    for i in range(6):
        img_0 = cv.imread('../data/goldengate-0'+str(i)+'.png')

        gray = cv.cvtColor(img_0, cv.COLOR_BGR2GRAY)

        kp, des = sift.detectAndCompute(gray, None)
        img_1 = cv.drawKeypoints(gray, kp, img_0)
        cv.imwrite('../results/sift_keypoints_'+str(i)+'.png', img_1)

        index = []
        for point in kp:
            temp = (point.pt, point.size, point.angle, point.response, point.octave,point.class_id)
            index.append(temp)

        f = open('../results/sift_' + str(i) + '.txt', 'wb')
        f.write(cp.dumps(index))
        f.write(cp.dumps(des.tolist()))
        f.close()

    for i in range(5):

        img1 = cv.imread('../data/goldengate-0'+str(i)+'.png')
        img2 = cv.imread('../data/goldengate-0'+str(i+1)+'.png')

        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

        bf = cv.BFMatcher()
        matches = bf.match(descriptors_1, descriptors_2)
        matched_img =  cv.drawMatches(img1, keypoints_1, img2, keypoints_2, matches, img2, flags=2)

        cv.imwrite('../results/tentative correspondences'+str(i)+'-'+str(i+1)+'.png', matched_img)

        index = []
        for ma in matches:
            temp = (ma.distance, ma.imgIdx, ma.queryIdx, ma.trainIdx)
            index.append(temp)

        f = open('../results/tentative correspondences_' + str(i) + '-' + str(i+1) + '.txt', 'wb')
        f.write(cp.dumps(index))
        f.close()


exercise_1()
