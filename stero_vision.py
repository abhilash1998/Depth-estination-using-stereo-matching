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
#F,good_pts=ransac.fit(deepcopy(pts1),deepcopy(pts2),0.01)

#print("F",F)

F2=cv2.findFundamentalMat(pts_x,pts_y,cv2.FM_RANSAC,0.01,0.97)[0]


print("F2",F2)
#cv2.computeOrientation()
#Eh=K.T @ F @K
Eh2=K.T @ F2 @K
#u,s,vh=np.linalg.svd(Eh)
u2,s,vh2=np.linalg.svd(Eh2)



#Eh2=K.T @ F2 @K

#u2,s,vh2=np.linalg.svd(Eh2)



#E=u @ np.array([[1,0,0],[0,1,0],[0,0,0]]) @ vh

E2=u2 @ np.array([[1,0,0],[0,1,0],[0,0,0]]) @ vh2
#pts,R2,T2,mask_pose=cv2.recoverPose(E2,deepcopy(pts1),deepcopy(pts2),K)
#print("R2,T2",R2,T2)
R,T=epipole.extracting_rot_trans(E2)

no_of_count=[]

for i in range(4):
    #print("T",T[i])
    no_of_count.append(epipole.linear_tri(deepcopy(pts1), deepcopy(pts2), R[i], T[i], K))

real_rot=no_of_count.index(max(no_of_count))

R_real=R[real_rot]
T_real=T[real_rot]
print("R",R_real)
print("T",T_real)
cv2.imshow("keypoints1_image",keypoints1_image)
cv2.imshow("keypoints2_image",keypoints2_image)

#T_real=T2.reshape(3)


r1=T_real/np.linalg.norm(T_real)
r2=np.array([-T_real[1],T_real[0],0])/np.sqrt(np.square(T_real[:1])).T
r3=np.cross(r1,r2)
R_rect=np.array([r1.T,r2.T,r3.T]).T
#wp_image=cv2.warpPerspective(image2,R2,(image2.shape[:2]))
Xy=np.vstack((pts2[:,0],pts2[:,1],np.ones(len(pts2)))).T
pts2_h=np.zeros(Xy.shape)



for i in range(len(Xy)):
    pts2_h[i,:]= ((K @ R_real @ np.linalg.inv(K))@(R_rect )@ Xy[i,:])

    pts2_h[i,:]=pts2_h[i,:]/pts2_h[i,-1]
#pts1_h=pts1_h/pts1_h[:,3]
pts2_h=pts2_h[:,:2]
#H2,r=cv2.findHomography(pts2,pts2_h)
#H1=K@R2@np.linalg.inv(K)

_,_,v=np.linalg.svd(F2
                )

e1=v[2,:]

e1=e1/e1[2]

_,_,v=np.linalg.svd(F2.T
                    )
e2=v.T[:,2]
e2=e2/e2[2]



def drawlines(img1,img2,lines,pts1,pts2):
    ''' Draw lines on the images using the features on image

    Parameters
    ----------
    img1 : Img
        input left image.
    img2 : Img
        Input right image.
    lines : np.array
        epipolar lines.
    pts1 : np.array
        feature points.
    pts2 : np.array
        feature points.

    Returns
    -------
    img1 : image
        image with lines plotted.
    img2 : TYPE
        image with lines plotted

    '''
    sh = img1.shape
    r = sh[0]
    c = sh[1]
    #img1 = cv2.cvtColor(img1,cv.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        pt1 = [int(pt1[0]),int(pt1[1])]
        pt2 = [int(pt2[0]),int(pt2[1])]
        #print("pt1 ",pt1)
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),2,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),2,color,-1)
    return img1,img2




#cv2.stereoRectify()
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F2)
lines1 = lines1.reshape(-1,3)
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F2)
lines2 = lines2.reshape(-1,3)


H1,H2=epipole.homography_R(e2,F2.T, image1, pts_y, pts_x)

image1_H1=cv2.warpPerspective(image3,np.linalg.inv(H1),(int(2988),int(2008)))
image_straight=cv2.warpPerspective(image4,np.linalg.inv(H2),(int(2988),int(2008)))

print("H1",H1)

print("H2",H2)
image3, image4 = drawlines(image1_H1,image4,lines1,pts1[:10],pts2[:10])
image3, image4 = drawlines(image4,image3,lines2,pts2[:10],pts2[:10])




#image3_S, image4_S = drawlines(image3.copy(),image4.copy(),lines1,pts1[:10],pts2[:10])
#image3_S, image4_S = drawlines(image4.copy(),image3.copy(),lines2,pts2[:10],pts2[:10])




#H1,H2=epipole.homography_R(e2,F2.T, image1, pts2, pts1)


#image1_H1_S=cv2.warpPerspective(image3_S,np.linalg.inv(H1),(int(2988),int(2008)))
#image_H2_S=cv2.warpPerspective(image4_S,np.linalg.inv(H2),(int(2988),int(2008)))


e_p1=np.hstack((pts1,np.ones((len(pts1),1))))
e_p2=np.hstack((pts2,np.ones((len(pts1),1))))





#e_lines1= F1@e_p1.T  
#e_lines2= F2.T@e_p1.TS






cv2.imshow("image1_H",cv2.resize(image1_H1,(int(2988/2),int(2008/2))))

print(R_rect)






disparity_map=epipole.disparity(cv2.resize(image1_H1,(640,480)), cv2.resize(image_straight,(640,480)))

disparity_map[ disparity_map<1]=1
#disparity_map[ disparity_map<3]=3
depth=baseline*f/disparity_map

plt.imshow(depth,cmap='gray', interpolation='bilinear')
plt.title("depth_map_gray")
plt.savefig('depth_map_gray.png')
plt.show()

plt.imshow(depth,cmap='hot', interpolation='bilinear')
plt.title("depth_map_heatmap")
plt.savefig('depth_map_heatmap.png')
plt.show()
#plt.title("depth")
cv2.imshow("wp_image",cv2.resize(image_straight,(int(2988/2),int(2008/2))))
#cv2.warpPerspective()
#matched_img=cv2.drawMatchesKnn(image1,keypoints1,image2,keypoints2,feature,outImg=None)
#cv2.imshow("matched_img",matched_img)




cv2.waitKey(0)
cv2.destroyAllWindows()
