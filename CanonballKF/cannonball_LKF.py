# -*- coding: utf-8 -*-


# Multivariable LKF

""" Assume that a ball is fired from a cannon at angle Theta at a muzzle 
velocity V0 units/sec. Assume that we have camera that records the ball's  
position from the side every second. The positions measured by the camera 
have significant measurement error. Assume that we also have 
precise detectors in the ball that give the X- and Y-velocity every second.
"""



""" we know that
Vx(t)=V0_x
X(t)=X0+V0_x*t
Vy(t)=V0_y-g*t
Y(t)=Y0+V0_y*t-0.5*g*t*t

Converting these equations to discrete domain results in
 equation for system dynamics
X(n)=X(n-1)+V(n-1)*Delta_t  //hor position = x0 + hor velocity x delta t
Vx(n)=Vx(n-1) deceleration vertically
Vy(n)=Vy(n-1)-g*Delta_t 
Y(n)=Y(n-1)+Vy(n-1)*Delta_t-0.5*g*Delta_t*Delta_t

Hence, our state vector is S=[X(n), Vx(n),Y(n),Vy(n)]
A=[1 Delta_t 0 0
   0 1       0 0
   0 0       1 Delta_t
   0 0       0 1]
B=[0 0 0 0
   0 0 0 0
   0 0 1 0
   0 0 0 1]
u=[0, 0, -0.5*g*Delta_t*Delta_t, -g*Delta_t]

Since the measurements are drectly mapped to state,
H=[1 0 0 0
   0 1 0 0
   0 0 1 0
   0 0 0 1]

S0 is the guess for the ball's state. This is where we seperate the muzzle velocity 
into its X/Y components. X0=[0, V0*Cos(Theta), 500, V0*Sin(Theta)]. Y0 is set to
illustrate the filter behaviour.X0[hor pos, vel along hor dir, altitude of platform where cannon is placed, 
"""
import numpy as np
import matplotlib.pyplot as plt

V0=100                  # Initial velocity
Theta=np.pi/3.0         # Firing angle
Delta_t=0.1             # Time step
Tot_TimeSTEPS=144       # Total number of iterations
g=9.81                  # gravitational acceleration
Pos_Measurement_NoiseLevel=15   # Noise level in measuring positions by side 
# camera

A=np.array([[1,Delta_t,0,0],[0,1,0,0],[0,0,1,Delta_t],[0,0,0,1]],dtype=np.float64)
B=np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.float64)

U=np.array([0, 0, -0.5*g*Delta_t*Delta_t, -g*Delta_t],dtype=np.float64)

H=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.float64)

S=[[0, 100*np.cos(Theta),500,100*np.sin(Theta)]]    # Initial state estimation
# [X, Vx,Y,Vy]
# Initial estimation of state process covariance matrix
P=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.float64)

# State process noise covariance matrix. Since equations are exact predictions
# Q is taken as zero
Q=np.array([[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.float64)

# We assumed that the measurement process variance is 0.2 for all variables
R=np.array([[0.2,0,0,0],[0,0.2,0,0],[0,0,0.2,0],[0,0,0,0.2]],dtype=np.float64)

ListMeasuredX=[0]
ListMeasuredY=[500]

ListKalmanX=[0]
ListKalmanY=[500]

ListTrueX=[0]
ListTrueY=[0]

TrueX=0
TrueY=0
for t in range(1,Tot_TimeSTEPS,1):
    Spredicted=np.matmul(A,np.array(S[t-1]).transpose())+np.matmul(B,U)
    
    Ppredicted=np.matmul(np.matmul(A,P),A.transpose())+Q
    
    # Get measurements: X/Y position measurements are noisy, whereas Vx and Vy 
    # are  precise
    Vx=S[t-1][1]
    Vy=S[t-1][3]-g*Delta_t
    ##from physics equation
    TrueX=TrueX+Vx*Delta_t
    ListTrueX.append(TrueX)

    ##from camera measurement
    ## true x + gaussian noise
    MeasuredX=TrueX+np.random.normal(0,Pos_Measurement_NoiseLevel)
    ListMeasuredX.append(MeasuredX)
    
    TrueY=TrueY+Vy*Delta_t-0.5*g*Delta_t*Delta_t
    ##when the ball hits the ground
    if TrueY<0:
        TrueY=0
        
    ListTrueY.append(TrueY)
    MeasuredY=TrueY+ np.random.normal(0,Pos_Measurement_NoiseLevel)
    if MeasuredY <0:
        MeasuredY=0
        
    ListMeasuredY.append(MeasuredY)

    ##compute d error
    Innovation=np.array([MeasuredX,Vx,MeasuredY,Vy])-np.matmul(H,Spredicted)
    
    ICov=np.matmul(np.matmul(H,Ppredicted),H.transpose()) + R
    
    K=np.matmul(np.matmul(Ppredicted,H.transpose()),np.linalg.inv(ICov))

    ##use kalman bain to estimate the new state value
    Sn=Spredicted + np.matmul(K,Innovation)
    if Sn[2] < 0:   # if Cannon ball hits the ground
        Sn[2]=0
        
    S.append(Sn)

    ##update the estimated x and y pos as a result of kalman est
    ListKalmanX.append(S[t][0])
    ListKalmanY.append(S[t][2])

    ##compute new state process covariance matrix
    P=np.matmul((np.eye(4)-np.matmul(K,H)),Ppredicted)
    
  

plt.plot(ListMeasuredX,ListMeasuredY,'k',ListTrueX,ListTrueY,'g', \
         ListKalmanX,ListKalmanY,'y')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Position Measurement of a Cannon Ball in Flight with Kalman Filter')
plt.legend(('Measured Pos.','True Pos.','Kalman Pos.'))
plt.show()    
