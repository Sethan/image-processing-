import matplotlib.pyplot as plt
import os

image_output_dir = "image_processed"
os.makedirs(image_output_dir, exist_ok=True)


def save_im(imname, im, cmap=None):
    impath = os.path.join(image_output_dir, imname)
    plt.imsave(impath, im, cmap=cmap)


def greyscale(im):
    """ Converts an RGB image to greyscale
    greyi,j = 0.212Ri,j + 0.7152Gi,j + 0.0722Bi,j 
    Args:
        im ([type]): [np.array of shape [H, W, 3]]
    
    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    # YOUR CODE HERE
	#loops trough a copy of the image and changes the colour value using the formula
    returnIm=im.copy()
    for x in range(returnIm.shape[0]):
        for y in range(returnIm.shape[1]):
            returnIm[x,y,:]=0.212*returnIm[x,y,0]+0.7152*returnIm[x,y,1]+0.0722*returnIm[x,y,2]
    return returnIm


def inverse(im):
    """ Finds the inverse of the greyscale image
    
    Args:
        im ([type]): [np.array of shape [H, W]]
    
    Returns:
        im ([type]): [np.array of shape [H, W]]
    """    
     # YOUR CODE HERE
	 #loops trough a copy of the image and changes the colour value using the formula
    returnIm=im.copy()
    for x in range(returnIm.shape[0]):
        for y in range(returnIm.shape[1]):
            returnIm[x,y,:]=255-returnIm[x,y,:]

   
    return returnIm	 



if __name__ == "__main__":
    im = plt.imread("images/lake.jpg")
    im = greyscale(im)
    inverse_im = inverse(im)
    save_im("lake_greyscale.jpg", im, cmap="gray")
    save_im("lake_inverse.jpg", inverse_im, cmap="gray")
