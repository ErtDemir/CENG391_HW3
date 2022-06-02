# 260201059 Ertuğrul Demir
import numpy as np
import cv2 as cv
import pickle as cp


# Exercise 2
def exercise_2():
    for i in range(5):
        img0 = cv.imread('../data/goldengate-0'+str(i)+'.png')
        f = open('../results/sift_'+str(i)+'.txt', 'rb')
        index0 = cp.loads(f.read())
        keypoints_0 = []

        for point in index0:
            temp0 = cv.KeyPoint(x=point[0][0],y=point[0][1],size=point[1], angle=point[2], response=point[3], octave=point[4], class_id=point[5])
            keypoints_0.append(temp0)

        img1 = cv.imread('../data/goldengate-0'+str(i+1)+'.png')
        f = open('../results/sift_'+str(i+1)+'.txt', 'rb')
        index1 = cp.loads(f.read())
        keypoints_1 = []

        for point in index1:
            temp1 = cv.KeyPoint(x=point[0][0],y=point[0][1],size=point[1], angle=point[2], response=point[3], octave=point[4], class_id=point[5])
            keypoints_1.append(temp1)

        f = open('../results/tentative correspondences_'+str(i)+'-'+str(i+1)+'.txt', 'rb')
        tent0 = cp.loads(f.read())
        matches = []

        for point in tent0:
            temp_tent_0 = cv.DMatch(_distance = point[0], _imgIdx = point[1], _queryIdx = point[2], _trainIdx = point[3])
            matches.append(temp_tent_0)

        src_pts = np.float32([ keypoints_0[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints_1[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)

        # For a better view
        matchesMask = mask.ravel().tolist()
        h,w,a = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img1 = cv.polylines(img1, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)
        img2 = cv.drawMatches(img0,keypoints_0,img1,keypoints_1,matches,None,**draw_params)

        mat = np.matrix(M)
        with open('../results/h_'+str(i)+'-'+str(i+1)+'.txt','w') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.2f')
        f.close()
        cv.imwrite('../results/inliers_'+str(i)+'-'+str(i+1)+'.png',img2)

        with open('../results/inliers_'+str(i)+'-'+str(i+1)+'.txt','w') as f: # Write
            for k in mask:
                for l in k:
                    f.write(str(l))


exercise_2()
