import numpy as np
import os
import matplotlib.pyplot as plt
from task2ab import save_im


def convolve_im(im, kernel):
    """ A function that convolves im with kernel
    
    Args:
        im ([type]): [np.array of shape [H, W, 3]]
        kernel ([type]): [np.array of shape [K, K]]
    
    Returns:
        [type]: [np.array of shape [H, W, 3]. should be same as im]
    """
    # YOUR CODE HERE
	"""
	first it saves the array dimensions and the offset (distance from a border to center) an creates a copy
	looping trough the image and for each pixel and calculating the summation of each kernel and corresponding image combination 
	the sum is saved in the copied versions corresponding pixel
	then the copy is returned
	
	"""
    arrayDim=len(kernel[0])
    offset=np.ceil(arrayDim/2)
    offset=int(offset)
    returnIm=im.copy()
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            sum=[0,0,0]
            for kx in range(arrayDim):
                for ky in range(arrayDim):
                    cX=x-kx+offset
                    cY=y-ky+offset
                    if(0<=cX and cX<im.shape[0] and 0<=cY and cY <im.shape[1]):
                        sum+=im[cX,cY]*kernel[kx,ky]
            returnIm[x,y]=sum					
    return returnIm


if __name__ == "__main__":
    # Read image
    impath = os.path.join("images", "lake.jpg")
    im = plt.imread(impath)

    # Define the convolutional kernels
    h_a = np.ones((3, 3)) / 9
    h_b = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    # Convolve images
    smoothed_im1 = convolve_im(im.copy(), h_a)
    smoothed_im2 = convolve_im(im, h_b)

    # DO NOT CHANGE
    assert isinstance(smoothed_im1, np.ndarray), \
        f"Your convolve function has to return a np.array. " +\
        f"Was: {type(smoothed_im1)}"
    assert smoothed_im1.shape == im.shape, \
        f"Expected smoothed im ({smoothed_im1.shape}" + \
        f"to have same shape as im ({im.shape})"
    assert smoothed_im2.shape == im.shape, \
        f"Expected smoothed im ({smoothed_im1.shape}" + \
        f"to have same shape as im ({im.shape})"

    save_im("convolved_im_h_a.jpg", smoothed_im1)
    save_im("convolved_im_h_b.jpg", smoothed_im2)
