# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 12, 2022
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals, estimateShape, plotSurface 
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface 

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals
    
    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    u, sigma, vh = np.linalg.svd(I,full_matrices=False)
    sigma[3:]=0
    B = vh[:3,:]
    L = u[:,:3]
    return B, L.T

def plotBasRelief(B, mu, nu, lam):

    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter
    
    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """
    G = np.asarray([[1,0,0],[0,1,0],[mu,nu,lam]])
    #print(G)
    new_B_est = np.linalg.inv(G.T) @ B_est
    #when inverted
    # x = np.eye(3)
    # x[-1,-1] = -1
    # new_B_est = np.linalg.inv(x) @ new_B_est

    albedos_est, normals_est = estimateAlbedosNormals(new_B_est)
    normals_est = enforceIntegrability(normals_est,s)
    albedoIm_est, normalIm_est = displayAlbedosNormals(albedos_est, normals_est, s)

    surface = estimateShape(normals_est, s)
    plotSurface(surface)
    

if __name__ == "__main__":

    # Part 2 (b)
    # print(1)
    I, L, s = loadData('C:/Users/sahar/Desktop/Acads/CVB-Spring22/hw6/hw6/data/')
    B_est, L_est = estimatePseudonormalsUncalibrated(I)
    # print(B_est)
    albedos_est, normals_est = estimateAlbedosNormals(B_est)
    albedoIm_est, normalIm_est = displayAlbedosNormals(albedos_est, normals_est, s)
    # plt.imshow(albedoIm_est.clip(0,0.5), cmap = 'gray')
    # plt.show()
    plt.imsave('2b-a.png', albedoIm_est.clip(0,0.5), cmap = 'gray')
    plt.imsave('2b-b.png', normalIm_est, cmap = 'rainbow')

    #print("Original L: \n", L)
    #print("Estimated L: \n", L_est)

    # Part 2 (d)
    surface = estimateShape(normals_est, s)
    #plotSurface(surface)

    # Part 2 (e)
    albedos_est, normals_est = estimateAlbedosNormals(B_est)
    normals_est = enforceIntegrability(normals_est,s)
    albedoIm_est, normalIm_est = displayAlbedosNormals(albedos_est, normals_est, s)

    surface = estimateShape(normals_est, s)
    #plotSurface(surface)

    # Part 2 (f)
    mu = 1
    gam = 1
    lam = 1
    plotBasRelief(B_est, mu, gam, lam)