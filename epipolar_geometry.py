# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:31:11 2021

@author: abhil
"""

import numpy as np
import matplotlib.pyplot as plt
class epipolar_geometry:
    def extracting_rot_trans(self,E):
        """
        This extracts the 4 rotation and translation given Essential Matrix

        Parameters
        ----------
        E : np.array
            Essential Matrix.

        Returns
        -------
        R : np.array
            Rotation matrix between 2 images.
        T : np.array
            Translation matrix between 2 images.

        """
        W=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        U,S,V=np.linalg.svd(E)
        R1=U@W@V
        T1=U[:,2]
        R2=U@W@V
        T2=-U[:,2]
        R3=U@(W.T)@V
        T3=U[:,2]        
        R4=U@(W.T)@V
        T4=-U[:,2]
        R1=R1/R1[-1,-1]
        R2=R2/R2[-1,-1]
        R3=R3/R3[-1,-1]
        R4=R4/R4[-1,-1]
        R=[R1,R2,R3,R4]
        T=[T1,T2,T3,T4]
        for i in range(len(R)):
            if np.linalg.det(R[i])<0:
                R[i]=-R[i]
                T[i]=-T[i]
        return R,T
    def obtain_3d(self,R,C,pts1,pts2,K):
        """
        Compute 3d point given rotation and image center of the feature points

        Parameters
        ----------
        R : np.array
            Rotation matrix
        C : np.array
            image center.
        pts1 : np.array
            Feature points.
        pts2 : np.array
            Feature points of image2.
        K : np.array
            Intrinsic Paramteres.

        Returns
        -------
        pts : np.array
            3d points.

        """
        M1=K @ np.hstack((np.identity(3),np.zeros(3).reshape(3,1)))
        #print(-R@C.reshape(3,1))
        M2=K @ np.hstack((R,-R@C.reshape(3,1)))
        
        #for i in range(2):
        pts=[]
        for i in range(len(pts1)):
            P=[]
            #print(M1[2,:])
            P0=pts1[i,0] * (M1[2,:]-M1[0,:])
            P1=pts1[i,1] * M1[2,:]-M1[1,:]
            P2=pts2[i,0] * M2[2,:]-M2[0,:]
            P3=pts2[i,1] * M2[2,:]-M2[1,:]
            P=[P0,P1,P2,P3]
            
            
            U,S,V=np.linalg.svd(P)
            V=V[-1]
            V=V/V[3]
            V=V[:3]
            pts.append(V)
        
        return np.array(pts)
    def linear_tri(self,pts1,pts2,R,T,K):
        """
        Given points and sets of rotation and translation
        calculates the best set of rotation and translation
        by chirality check

        Parameters
        ----------
        pts1 : np.array
            Feature points of image1.
        pts2 : TYPE
            DESCRIPTION.
        R : TYPE
            DESCRIPTION.
        T : TYPE
            DESCRIPTION.
        K : TYPE
            DESCRIPTION.

        Returns
        -------
        counter : TYPE
            DESCRIPTION.

        """
        counter=0
        X=self.obtain_3d(R,T,pts1,pts2,K)
        for i in range(len(X)):
            #print(X[i,:])
            if R[2,:]@(X[i,:]-T)>0 and X[i,2]>0:
                counter=counter+1
        return counter
    def homography_R(self,e2,F,image1,pts2,pts1):
        """
        Calculates homography for image 1 and image 2 which will rectify the image
        making the epipole go to infinity

        Parameters
        ----------
        e2 : np.array
            epipole
        F : np.array
            Fundamental matrix.
        image1 : img
            Image1.
        pts2 : np.array
            feature points for image 2.
        pts1 : np.array
            feature points for image1.

        Returns
        -------
        H1 :np.array 
            Homography for image 1
            
        H2 : np.array
            homography for image 2.

        """
        w,h=image1.shape
        e_skew = np.asarray([ [0, -e2[2], e2[1] ],
                           [e2[2], 0 , -e2[0]],
                           [-e2[1], e2[0], 0 ] ])
        v = np.array([1, 1, 1])
        T1=np.array([[1,0,-w/2],[0,1,-h/2],[0,0,1]])
        e=T1@e2
        e_hyp=np.sqrt(e[0]**2 + e[1]**2)
        pts1=np.vstack((pts1.T,np.ones(len(pts1)))).T
        pts2=np.vstack((pts2.T,np.ones(len(pts2)))).T
        
        #theta=np.arctan2(e2[1],e2[0])
        
        T2=np.array([[e[0]/e_hyp,e[1]/e_hyp,0],[-e[1]/e_hyp,e[0]/e_hyp, 0],[0,0,1]])
        #T2=np.array([[np.cos(-theta),-np.sin(-theta),0],[np.sin(-theta),np.cos(-theta), 0],[0,0,1]])
        e=T2@e
        T3=np.identity(3)
        #e=e/e[2]
        #np.linalg.n
        
        T3[-1][0]=-1/e[0]
        
        #H2=T3@T2@T1
        H2=np.linalg.inv(T1)@T3@T2@T1
        #e12_new=H2@e2
        #H2=H2/e12_new[0]
        #e12_new=H2@e12_new
        
        

        
        

        
        M = (e_skew @ (F) + np.outer(e2, v))
        
        pts_H2=(H2@(pts2.T)).T
        pts_H1=(H2@M@pts1.T).T
        pts_H1=pts_H1/(pts_H1[:,-1]).reshape(-1,1)
        pts_H2=pts_H2/(pts_H2[:,-1]).reshape(-1,1)
        pts_H2=pts_H2[:,0]
        
        alpha=np.linalg.lstsq(pts_H1,pts_H2)
        alpha=alpha[0]
        
        
        alpha=np.array(alpha)
        
        a=np.identity(3)
        
        a[0]=alpha
        H1=(a)@H2@M
        
        return H1,H2
    def SSD(self,window_l, window_r):
        """
        Calculates sum of square differences

        Parameters
        ----------
        window_l : np.array
            left window.
        window_r : np.array
            right window.

        Returns
        -------
        
            SSD

        """
        if window_l.shape != window_r.shape:
            return -np.inf
    
        return np.sum(np.square(window_l - window_r))




    def compare_window(self,y, x, left_window, right_img, block_size=5):
        """
        
        Compares the value off SSD and returns the index for minimum SSd 

        """
        neglect_window_size = 49
        # Get search range for the right image
        x_min = max(0, x - neglect_window_size)
        x_max = min(right_img.shape[1], x + neglect_window_size)
        #print(f'search bounding box: ({y, x_min}, ({y, x_max}))')
        
        mini = np.inf
        mini_index = (0,0)
        for x in range(x_min, x_max):
            right_window = right_img[y: y+block_size, x: x+block_size]
            ssd = self.SSD(left_window, right_window)
            
            if ssd < mini:
                mini = ssd
                mini_index = (y, x)
    
        return mini_index
    
    
    
    
    def disparity(self,img_l,img_r):
        """
        Given images calculates disparity using SSD

        Parameters
        ----------
        img_l :img
            Left rectified image
            .
        img_r : img
            right rectified image.

        Returns
        -------
        disparity_map : np.array
            Disparity map.

        """
        WINDOW_SIZE = 7
        
        #array_l = np.asarray(img_l)
        #array_r = np.asarray(img_r)
        #array_l = array_l.astype(int)
        #array_r = array_r.astype(int)
        array_l=img_l.astype(int)
        array_r=img_r.astype(int)
        
        m, n , _ = array_l.shape
        
        disparity_map = np.zeros((m, n))
        
        for i in range(WINDOW_SIZE, m-WINDOW_SIZE):
            for j in range(WINDOW_SIZE, n-WINDOW_SIZE):
                block_left = array_l[i:i + WINDOW_SIZE,
                                        j:j + WINDOW_SIZE]
                ind_min = self.compare_window(i, j, block_left,
                                           array_r,
                                           block_size=WINDOW_SIZE)
                disparity_map[i, j] = abs(ind_min[1] - j)
    
    
        print(disparity_map)
        plt.imshow(disparity_map, cmap='gray', interpolation='bilinear')
        plt.title("disparity_map_gray")
        plt.savefig('disparity_map_gray.png')
        plt.show()
        
        plt.imshow(disparity_map, cmap='hot', interpolation='nearest')
        plt.title("disparity_map_heatmap")
        plt.savefig('disparity_map_heatmap.png')
        plt.show()
        
        return disparity_map
    
