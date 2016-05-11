import numpy as np
import cv2

def projectData(X,U,K):
    X = np.mat(X)
    U = np.mat(U)
    z = X*U[:,0:K]
    return z

def normalize(X,mu,sigma):
    return (X-mu)/sigma

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoidGradient(z):
    return np.multiply(sigmoid(z),1-sigmoid(z))

def appenOne(X):
    output = np.ones((X.shape[0],X.shape[1]+1))
    output[:,1:] = X
    return output

def getProbability(X,Theta1,Theta2,Theta3):
    m = np.shape(X)[0]
    X = np.mat(X)
    h1 = sigmoid(appenOne(X)*Theta1.transpose())
    h2 = sigmoid(appenOne(h1)*Theta2.transpose())
    h3 = sigmoid(appenOne(h2)*Theta3.transpose())
    return h3.tolist()[0]

def convertImg(im,mu,sigma):
    im_new = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im_new = im_new.reshape(1,im_new.size)
    im_new = normalize(im_new,mu,sigma)
    return im_new

def randomWeight(L_in, L_out):
    epsilon_init = 0.12
    return np.random.rand(L_out, L_in+1)*2*epsilon_init - epsilon_init

def initNNparams(input_layer_size,hidden_layer_size1,hidden_layer_size2,num_labels):
    param1 = randomWeight(input_layer_size, hidden_layer_size1)
    param2 = randomWeight(hidden_layer_size1, hidden_layer_size2)
    param3 = randomWeight(hidden_layer_size1, num_labels)
    return np.append(np.append(param1.reshape((1,param1.size),order='F'),param2.reshape((1,param2.size),order='F')),param3.reshape((1,param3.size),order='F'))

def costFunction(nn_params, input_layer_size, hidden_layer_size1, hidden_layer_size2, num_labels, X, y, lmd):
    ct = 0
    Theta1 = np.reshape(nn_params[0:hidden_layer_size1*(input_layer_size+1)],(hidden_layer_size1,(input_layer_size+1)),order='F')
    ct += hidden_layer_size1*(input_layer_size+1)
    Theta2 = np.reshape(nn_params[ct:(ct+hidden_layer_size2*(hidden_layer_size1+1))],(hidden_layer_size2,(hidden_layer_size1+1)),order='F')
    ct += hidden_layer_size2*(hidden_layer_size1+1)
    Theta3 = np.reshape(nn_params[ct:(ct+num_labels*(hidden_layer_size2+1))],(num_labels,(hidden_layer_size2+1)),order='F')
    m = np.shape(X)[0]
    J = 0.0
    Theta1_grad = np.zeros(np.shape(Theta1))
    Theta2_grad = np.zeros(np.shape(Theta2))
    Theta3_grad = np.zeros(np.shape(Theta3))
    Theta1 = np.mat(Theta1)
    Theta2 = np.mat(Theta2)
    Theta3 = np.mat(Theta3)
    Theta1_grad = np.mat(Theta1_grad)
    Theta2_grad = np.mat(Theta2_grad)
    Theta3_grad = np.mat(Theta3_grad)
    z2 = np.append(np.ones((m,1)),X,1)*Theta1.transpose()
    a2 = sigmoid(z2)
    z3 = np.append(np.ones((m,1)),a2,1)*Theta2.transpose()
    a3 = sigmoid(z3)
    z4 = np.append(np.ones((m,1)),a3,1)*Theta3.transpose()
    hp = sigmoid(z4)

    for i in range(0,num_labels):
        lb = i+1
        yVec = (y==lb)
        J += np.sum(-np.multiply(yVec,np.log(hp[:,i]))-np.multiply(1.0-yVec,np.log(1.0-hp[:,i])))/m


    J += (np.sum(np.square(Theta1[:,2:]))+np.sum(np.square(Theta2[:,2:]))+np.sum(np.square(Theta3[:,2:])))*lmd/(2*m)

    for exm in range(0,m):
        a1_exm = np.mat(np.append(1,X[exm,:]))
        a2_exm = np.mat(np.append(1,a2[exm,:]))
        a3_exm = np.mat(np.append(1,a3[exm,:]))
        z2_exm = z2[exm,:]
        z3_exm = z3[exm,:]
        hp_exm = hp[exm,:]
        Delta4 = hp_exm.transpose() - (np.mat(range(1,num_labels+1))==y[exm]).transpose()
        Delta3 = np.multiply(Theta3.transpose()*Delta4,sigmoidGradient(np.append(np.zeros((1,1)),z3_exm.transpose(),0)))
        Delta3 = Delta3[1:]
        Delta2 = np.multiply(Theta2.transpose()*Delta3,sigmoidGradient(np.append(np.zeros((1,1)),z2_exm.transpose(),0)))
        Delta2 = Delta2[1:]
        Theta1_grad += Delta2*a1_exm
        Theta2_grad += Delta3*a2_exm
        Theta3_grad += Delta4*a3_exm
    theta1Box = np.append(np.zeros((Theta1.shape[0],1)),Theta1[:,1:],1)
    theta2Box = np.append(np.zeros((Theta2.shape[0],1)),Theta2[:,1:],1)
    theta3Box = np.append(np.zeros((Theta3.shape[0],1)),Theta3[:,1:],1)

    Theta1_grad = Theta1_grad/m+theta1Box*lmd/m
    Theta2_grad = Theta2_grad/m+theta2Box*lmd/m
    Theta3_grad = Theta3_grad/m+theta3Box*lmd/m

    grad = np.append(np.array(Theta1_grad.reshape((1,Theta1_grad.size),order='F')),Theta2_grad.reshape((1,Theta2_grad.size),order='F'))
    grad = np.append(grad,Theta3_grad.reshape((1,Theta3_grad.size),order='F'))

    return J,grad