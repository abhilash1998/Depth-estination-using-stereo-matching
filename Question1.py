# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 22:20:21 2021

@author: abhil
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 20:46:51 2021

@author: abhil
"""

import cv2
import numpy as np
from ransac import Ransac
from copy import deepcopy
from epipolar_geometry import epipolar_geometry
import matplotlib.pyplot as plt

import os
file_location=os.getcwd()  




ransac=Ransac()
epipole=epipolar_geometry()

#image1=cv2.imread(r"C:/Users/abhil/OneDrive/Desktop/ENPM673/project3/Dataset 1/im0.png")
#image2=cv2.imread(r"C:/Users/abhil/OneDrive/Desktop/ENPM673/project3/Dataset 1/im1.png")

image1=cv2.imread(file_location+"/Dataset 1/im0.png")#For dataset 1
image2=cv2.imread(file_location+"/Dataset 1/im1.png")   


#image1=cv2.imread(file_location+"/Dataset 2/im0.png")#For dataset 2
#image2=cv2.imread(file_location+"/Dataset 2/im1.png")


#image1=cv2.imread(file_location+"/Dataset 3/im0.png")# For dataset 3
#image2=cv2.imread(file_location+"/Dataset 3/im1.png")





K=np.array([[5299.313, 0, 1263.818],[ 0 ,5299.313, 977.763],[ 0, 0, 1]])#For dataset 1
#K=np.array([[4396.869,0,1353.072],[ 0,4396.869,989.702],[ 0, 0, 1]])#For dataset 2
#K=np.array([[5806.559 ,0, 1429.219],[ 0 ,5806.559 ,993.403],[ 0 ,0 ,1]])# For dataset 3

baseline=177.288    # dataset1
f=5299.313

#baseline=144.049#dataset2
#f=4396.869

#baseline=174.019#dataset3
#f=5806.559





image3=image1.copy()
image4=image2.copy()

cv2.imshow("image1",cv2.resize(image1,(640,480)))
cv2.imshow("image2",image2)
image1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
image2=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
cv2.imshow("image1_g",image1)
cv2.imshow("image2_g",image2)


feature=[]

sift_object=cv2.xfeatures2d.SIFT_create()
keypoints1,descriptor1=sift_object.detectAndCompute(image1,None)
keypoints2,descriptor2=sift_object.detectAndCompute(image2,None)
#orb=cv2.ORB_create()
#keypoints1,descriptor1=orb.detectAndCompute(image1,None)
#keypoints2,descriptor2=orb.detectAndCompute(image2,None)

#K=np.array([[5299.313, 0, 1263.818],[ 0 ,5299.313, 977.763],[ 0, 0, 1]])


#baseline=177.288

#f=5299.313

keypoints1_image=cv2.drawKeypoints(image1,keypoints1,None,color=(0,255,0))
keypoints2_image=cv2.drawKeypoints(image2,keypoints2,None,color=(0,255,0))


match=cv2.BFMatcher()
matches=match.knnMatch(descriptor1, descriptor2, 2)

for im1,im2 in matches:
    if im1.distance< 0.75*im2.distance:
        feature.append(im1)
pts1=[]
pts2=[]
pts_x=[]
pts_y=[]

for  i,feat in enumerate(feature):
    pts1.append(list(keypoints1[feat.queryIdx].pt))
    pts2.append(list(keypoints2[feat.trainIdx].pt))
    pts_x.append(list(keypoints1[feat.queryIdx].pt))
    pts_y.append(list(keypoints2[feat.trainIdx].pt))
pts1=np.asarray(pts1)
pts2=np.asarray(pts2)
pts_x=np.array(pts_x)
pts_y=np.array(pts_y)
F,good_pts=ransac.fit(deepcopy(pts1),deepcopy(pts2),0.01)

print("F",F)





#cv2.computeOrientation()
Eh=K.T @ F @K

u,s,vh=np.linalg.svd(Eh)




#Eh2=K.T @ F2 @K

#u2,s,vh2=np.linalg.svd(Eh2)



E=u @ np.array([[1,0,0],[0,1,0],[0,0,0]]) @ vh
print("E",E)



R,T=epipole.extracting_rot_trans(E)

no_of_count=[]
for i in range(4):
    #print("T",T[i])
    no_of_count.append(epipole.linear_tri(deepcopy(pts1), deepcopy(pts2), R[i], T[i], K))

real_rot=no_of_count.index(max(no_of_count))

R_real=R[real_rot]
T_real=T[real_rot]
print("R_true",R_real)
print("T_true",T_real)
cv2.imshow("keypoints1_image",keypoints1_image)
cv2.imshow("keypoints2_image",keypoints2_image)

#T_real=T2.reshape(3)


#wp_image=cv2.warpPerspective(image2,R2,(image2.shape[:2]))

cv2.waitKey(0)
cv2.destroyAllWindows()
