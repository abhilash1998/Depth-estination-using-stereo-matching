# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:14:52 2021

@author: abhil
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:35:46 2021

@author: abhi
"""

import numpy as np

"""
This class computes the ransac
Input - Null constant vector that needs to be found for curve fitting and coordinates
Output - Constants of the equation representing curve fitting the data by RANSAC
Selects Random 3 points gets an equation of the curve count the inliers and iterates
untill the maximum number of inlier is achieved and returns the constant of the equation
 for which the curve will best fit the data points  """
class Ransac:

    def eqn_formation(self,coordinates1,coordinates2):
        """
        This forms the equation of

        Parameters
        ----------
        coordinates1 : np.array
            Coordinates features pts 1 .
        coordinates2 : np.array
            Feature pts 2

        Returns
        -------
        F : np.array
            Fundamental Matrix of the image
            .

        """
        A=np.zeros((8,9))
        #compute the centroids
        coordinates11=coordinates1.copy()
        coordinates21=coordinates2.copy()
        
        
        m1 = np.mean(coordinates1[:, 0])
        m2 = np.mean(coordinates1[:, 1])
        n1 = np.mean(coordinates2[:, 0])
        n2 = np.mean(coordinates2[:, 1])

        #Recenter the coordinates by subtracting mean
        coordinates11[:,0] = coordinates1[:,0] - m1
        coordinates11[:,1] = coordinates1[:,1] - m2
        coordinates21[:,0] = coordinates2[:,0] - n1
        coordinates21[:,1] = coordinates2[:,1] - n2


        meanm=np.array([m1,m2])
        meann=np.array([n1,n2])




        
        s1 = np.sqrt(2.)/np.mean(np.sum((coordinates11)**2,axis=1)**(1/2))
        s2 = np.sqrt(2.)/np.mean(np.sum((coordinates21)**2,axis=1)**(1/2))


        
        T_a_1 = np.array([[s1,0,0],[0,s1,0],[0,0,1]])
        T_a_2 = np.array([[1,0,-m1],[0,1,-m2],[0,0,1]])
        T_a = T_a_1 @ T_a_2

        
        T_b_1 = np.array([[s2,0,0],[0,s2,0],[0,0,1]])
        T_b_2 = np.array([[1,0,-n1],[0,1,-n2],[0,0,1]])
        T_b = T_b_1 @ T_b_2
        
        T2=T_a
        T1=T_b
        self.p1=((coordinates1)-meanm)


        self.p2=((coordinates2)-meann)
        
        self.p1=((coordinates1)-meanm)


        self.p2=((coordinates2)-meann)
        
        self.x1=self.p1[:,0]*s1
        self.y1=self.p1[:,1]*s1
        self.x2=self.p2[:,0]*s2
        self.y2=self.p2[:,1]*s2

        


            
        A=A.T
        A[0]=self.x2*self.x1
        A[1]=self.x2*self.y1
        A[2]= self.x2
        A[3]=self.x1*self.y2
        A[4]=self.y2*self.y1
        A[5]= self.y2
        A[6]=self.x1
        A[7]=self.y1
        A[8]= 1
        A=A.T
        u,s,vh=np.linalg.svd(A,full_matrices=True)
        vh=vh.T
        F=(vh[:,-1]
           ).reshape(3,3)

        U,S,V=np.linalg.svd(F)
        S[-1]=0

        S=np.diag(S)

        F=((U@S)@V)
        #F=(np.linalg.inv(T2).T)@F@np.linalg.inv(T1)
        F=((T2).T)@F@(T1)
        

        F=F/F[-1,-1]
        return F


    
    def fit(self,coordinates1,coordinates2,threshold):
        """
        Calculates the 

        Parameters
        ----------
        coordinates1 : TYPE
            DESCRIPTION.
        coordinates2 : TYPE
            DESCRIPTION.
        threshold : TYPE
            DESCRIPTION.

        Returns
        -------
        A_fin : TYPE
            DESCRIPTION.
        good_pts : TYPE
            DESCRIPTION.

        """
        #RANSAC parameters
        threshold =0.05
        max_iter=np.inf
        no_of_iter=0
        max_inlier=0
        p=0.99
        e=0.5
        A_fin=np.zeros(3)
        good_pts=np.zeros(coordinates1.shape)
        while max_iter>no_of_iter:

            coord1 = []
            coord2 = []
            
            rand_index = np.random.randint(len(coordinates1), size = 8)
            for i in rand_index:
                coord1.append(coordinates1[i])
                coord2.append(coordinates2[i])
            
            #np.random.shuffle(coordinates1)
            #np.random.shuffle(coordinates2)
            #random_8_coordinates1=coordinates1[-8:]
            #random_8_coordinates2=coordinates2[-8:]
            coord1=np.array(coord1)
            coord2=np.array(coord2)
            F = self.eqn_formation(coord1, coord2)
            ones = np.ones((len(coordinates1),1))
            p1=np.vstack((coordinates1[:,0],coordinates1[:,1], np.ones(len(coordinates1)))).T
            p2=np.vstack((coordinates1[:,0],coordinates1[:,1], np.ones(len(coordinates1)))).T
            #x1 = np.hstack((coordinates1,ones))
            #x2 = np.hstack((coordinates2,ones))
            e0= p1 @ F.T
            e1= p2 @ F
            y = np.sum(e1* p1, axis = 1, keepdims=True)**2 / np.sum(np.hstack((e0[:, :-1],e1[:,:-1]))**2, axis = 1, keepdims=True)
            y[y<=threshold]=0
            num_inlier = np.count_nonzero(y)
            e=1-(num_inlier/len(coordinates1))

            if max_inlier<num_inlier:
                max_inlier=num_inlier
                
                #good_pts=coordinates1[y==0,1]
                
                A_fin=F

            """  adaptive thresholding"""
            #print(e)
            max_iter=np.log(1-p)/np.log(1-(1-e)**(8))
            #print(np.log(1-(1-e)**(3)))

            no_of_iter+=1
        print(good_pts)
        return A_fin,good_pts
