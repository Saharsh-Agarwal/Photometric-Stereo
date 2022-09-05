# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 12, 2022
###################################################################### #

import numpy as np
from matplotlib import pyplot as plt
from pyparsing import NoMatch
import skimage
from skimage.color import rgb2xyz
from utils import integrateFrankot, plotSurface
import os

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a hemispherical bowl with a given center and radius. Assume that
    the hollow end of the bowl faces in the positive z direction, and the
    camera looks towards the hollow end in the negative z direction. The
    camera's sensor axes are aligned with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray #0,0,0
        The center of the hemispherical bowl in an array of size (3,)

    rad : float #rad of sphere 0.75
        The radius of the bowl

    light : numpy.ndarray #dir (1/root(3))
        The direction of incoming light

    pxSize :  #7
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame #3840,2160

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    ### print(center, rad, light, pxSize, res)
    [X, Y] = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    # print(X.shape, Y.shape) #2160, 3840
    X = (X - res[0]/2) * pxSize*1.e-4 # everything in cm
    Y = (Y - res[1]/2) * pxSize*1.e-4
    Z = np.sqrt(rad**2+0j-X**2-Y**2)
    X[np.real(Z) == 0] = 0
    Y[np.real(Z) == 0] = 0
    Z = np.real(Z)
    # print(Z[1080,1920])
    # plt.imshow(Z)
    # plt.show() ## sphere

    x = X.reshape(-1)
    y = Y.reshape(-1)
    z = Z.reshape(-1)
    pts = np.vstack((np.vstack((x,y)),z))
    
    dist = np.linalg.norm(pts,axis=0)+1.e-8 #to avoid 0/0

    # print(dist[36450])
    # norm_pts = pts
    # print(norm_pts.shape)
    # for i in range(3):
    #     print(i, norm_pts[i].shape, pts[i].shape,dist.shape)
    #     norm_pts[i] = pts[i]/dist #could have just divided the image by 0.75
    #print(norm_pts.shape,light.shape)

    n_pts = pts/dist
    image = np.dot(n_pts.T,light).reshape((res[1],res[0]))

    #print(image[1920])

    a = image<0
    image[a]=0
    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Parameters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    # loadData function
    I = None
    L = None
    s = None
    
    for i in range(7):
        pth = os.path.join(path,"input_" + str(i+1) + ".tif")
        print(pth)
        img = skimage.img_as_uint(skimage.io.imread(pth))
        #print(img.shape) #431,369,3
        c_img = rgb2xyz(img) 
        l_img = c_img[:,:,1] #Y channel for luminance
        if I is None:
            pcount = l_img.shape[0]*l_img.shape[1]
            I = np.zeros((7,pcount))
            s = l_img.shape
        I[i,:] = l_img.reshape((1,pcount)) #P=pcount

    L = np.load(os.path.join(path,"sources.npy")).T
    print(L.shape)

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = np.linalg.inv(L@L.T)@L@I
    #print(B.shape) #3,159039 (431*369)
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = np.linalg.norm(B, axis=0)
    #print(albedos.shape)#all the pixels - pcount
    normals = B/(albedos+1.e-8)
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f, g)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    mina = np.min(albedos)
    maxa = np.max(albedos)
    al_norm = (albedos-mina)/(maxa-mina)
    albedoIm = al_norm.reshape(s)
    # print(normals.shape) 3,pcount
    normalIm = (((normals+1)/2).T).reshape((s[0],s[1],3))

    minn = np.min(normalIm)
    maxn = np.max(normalIm)
    normalIm = (normalIm-minn)/(maxn-minn)
    
    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (j)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    print(normals.shape)

    z_x = -(normals[0,:]/(normals[2,:]+1.e-8)).reshape(s)
    z_y = -(normals[1,:]/(normals[2,:]+1.e-8)).reshape(s)
    surface = integrateFrankot(z_x,z_y)
    #print(surface.shape)
    return surface

if __name__ == '__main__':
    # Part 1(b)
    radius = 0.75 # cm
    center = np.asarray([0, 0, 0]) # cm
    pxSize = 7 # um
    res = (3840, 2160)

    light = np.asarray([1, 1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.imsave('1b-a.png', image, cmap = 'gray')
    #plt.show()

    light = np.asarray([1, -1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.imsave('1b-b.png', image, cmap = 'gray')
    #plt.show()

    light = np.asarray([-1, -1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.imsave('1b-c.png', image, cmap = 'gray')
    #plt.show()

    # Part 1(c)
    I, L, s = loadData('C:/Users/sahar/Desktop/Acads/CVB-Spring22/hw6/hw6/data/')
    
    # Part 1(d)
    u,sigma,vh = np.linalg.svd(I,full_matrices=False)
    print(sigma)

    # Part 1(e)
    B = estimatePseudonormalsCalibrated(I, L)

    # Part 1(f)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave('1f-a.png', albedoIm.clip(0,0.5), cmap = 'gray')
    plt.imsave('1f-b.png', normalIm, cmap = 'rainbow')

    # Part 1(i)
    print("s",s)
    surface = estimateShape(normals, s)
    plotSurface(surface)