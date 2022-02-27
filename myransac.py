# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 00:28:01 2021

@author: abhil
"""

def eqn_formation(self,coordinates1,coordinates2):
        A=np.zeros((8,9))
        
        """Randomly selects 8  points and plot the equaiton"""
        np.random.shuffle(coordinates1)
        np.random.shuffle(coordinates2)
        coordinates1=(coordinates1[-8:,:])
        coordinates2=(coordinates2[-8:,:])
        
        
        #print(coordinates1[1][:])
        m1=np.mean(coordinates1[:,0])
        m2=np.mean(coordinates1[:,1])
        
        n1=np.mean(coordinates2[:,0])
        n2=np.mean(coordinates2[:,1])
        meanm=np.array([m1,m2])
        meann=np.array([n1,n2])
        
        s1=np.mean(np.sum((coordinates1-meanm)**2)**(0.5))
        s2=np.mean(np.sum((coordinates2-meann)**2)**(0.5))
        
        s1=np.sqrt(2)/s1
        s2=np.sqrt(2)/s2
        
        #s1=np.linalg.norm(((coordinates1-meanm)))/(np.sqrt(2)*len(coordinates1[:,1]))
        #s2=np.linalg.norm(((coordinates2-meann)))/(np.sqrt(2)*len(coordinates2[:,1]))
    
        
        #s1=(np.sum((coordinates1[:,0]-np.mean(coordinates1[:,0]))**2+(coordinates1[:,1]-np.mean(coordinates1[:,1]))**2)/np.sqrt(2)*len(coordinates1[:,1]))
        #s2=(np.sum((coordinates2[:,0]-np.mean(coordinates2[:,0]))**2+(coordinates2[:,1]-np.mean(coordinates2[:,1]))**2)/np.sqrt(2)*len(coordinates2[:,1]))
    
        """self.x1=((coordinates1[-8:,0])-np.mean(coordinates1[:,0]))/s1
        
        
        self.y1=((coordinates1[-8:,1])-np.mean(coordinates1[:,1]))/s1
        self.x2=((coordinates2[-8:,0])-np.mean(coordinates2[:,0]))/s2
        self.y2=((coordinates2[-8:,1])-np.mean(coordinates2[:,1]))/s2
        
        
        T1=np.array([[1/s1,0,-m1/s1],[0 ,1/s1, -m2/s1],[0,0,1]])
        T2=np.array([[1/s2,0,-n1/s2],[0 ,1/s2, -n2/s2],[0,0,1]])
        """
        self.p1=((coordinates1)-meanm)
        
        
        self.p2=((coordinates2)-meann)
        #self.x2=((coordinates2[-8:,0])-np.mean(coordinates2[:,0]))*s2
        #self.y2=((coordinates2[-8:,1])-np.mean(coordinates2[:,1]))*s2
        self.x1=self.p1[:,0].reshape((-1,1))*s1
        self.y1=self.p1[:,1].reshape((-1,1))*s1
        self.x2=self.p2[:,0].reshape((-1,1))*s2
        self.y2=self.p2[:,1].reshape((-1,1))*s2
        T1=np.array([[s1,0,-meanm[0]*s1],[0 ,s1, -meanm[1]*s1],[0,0,1]])
        T2=np.array([[s2,0,-meann[0]*s2],[0 ,s2, -meann[1]*s2],[0,0,1]])
        
        A = np.hstack((self.x2*self.x1, self.x2*self.y1, self.x2, self.y2 * self.x1,self.y2 * self.y1, self.y2, self.x1, self.y1, np.ones((len(self.x1),1))))    
        """for i in range(8):
            
            #print(i)
            A[i,0]=self.x2[i]*self.x1[i]
            A[i,1]=self.x2[i]*self.y1[i]
            A[i,2]= self.x2[i]
            A[i,3]=self.x1[i]*self.y2[i]
            A[i,4]=self.y2[i]*self.y1[i]
            A[i,5]= self.y2[i]
            A[i,6]=self.x1[i]
            A[i,7]=self.y1[i]
            A[i,8]= 1
            """
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
        #F = T_b.T @ F @ T_a

        F=F/F[-1,-1]
        #print(F.shape)
        #X=(np.hstack((self.x**2,self.x,np.ones([3,1]))))
        #constant=(np.linalg.inv(X).dot(self.y))       
  
        return F
    def fit(self,coordinates1,coordinates2,threshold):
        """ Calculate the inliers and give the best equation out"""        
        max_iter=2000
        no_of_iter=0
        max_inlier=0
        p=0.99
        e=0.5
        A_fin=np.zeros(3)
        while max_iter>no_of_iter:
            
            F=self.eqn_formation(coordinates1.copy(),coordinates2.copy())
            
            #x_co=coordinates[:,1].reshape(len(coordinates[:,1]),1)
            #print(np.vstack((coordinates1[:,0],coordinates1[:,1], np.ones(len(coordinates1)))).T.shape)
            y=[]
            XX=np.vstack((coordinates1[:,0],coordinates1[:,1], np.ones(len(coordinates1))))
            YY=np.vstack((coordinates1[:,0],coordinates1[:,1], np.ones(len(coordinates1))))
            #for i in range(len(coordinates1)):
            #print(XX[:,i].shape)
            #y_h=XX[:,i].T @ F 
            e1, e2 = XX.T @ F.T, YY.T @ F
            #print(y_h.shape)
            y = np.sum(e2* XX.T, axis = 1, keepdims=True)**2 / np.sum(np.hstack((e1[:, :-1],e2[:,:-1]))**2, axis = 1, keepdims=True)
            #y.append(y_h @ XX[:,i])
            #x_coord=(np.hstack((x_co**2,x_co,np.ones([len(coordinates[:,1]),1]))))  
            #y_fin=x_coord.dot(A)
       
            #error_y=np.square((coordinates[:,0].reshape(len(coordinates[:,0]),1))-y_fin)
            y=np.array(y)
            #print("y.shape",y.shape)
            y[y>threshold]=0
            #error_y[error_y>threshold]=0
            num_inlier = np.count_nonzero(y)
            
            #print(num_inlier)
            e=1-(num_inlier/len(coordinates1))
          
            if max_inlier<num_inlier:
                max_inlier=num_inlier
                A_fin=F
           
            """  adaptive thresholding"""
            #print(e)
            max_iter=np.log(1-p)/np.log(1-(1-e)**(3))
            #print(np.log(1-(1-e)**(3)))
            
            no_of_iter+=1
         
        return A_fin