import skimage
import numpy as np
import utils


def MaxPool2d(im: np.array,
              kernel_size: int):
    """ A function that max pools an image with size kernel size.
    Assume that the stride is equal to the kernel size, and that the kernel size is even.

    Args:
        im: [np.array of shape [H, W, 3]]
        kernel_size: integer
    Returns:
        im: [np.array of shape [H/kernel_size, W/kernel_size, 3]].
    """
    stride = kernel_size
	
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    """
    first it creates a new array that is scaled with the kernel size
    then looping trough the entire new image, it checks a kernel size chunk of the original image
	for the largest value in the image for each color channel  
	then the new image gets that value added to itself
	after looping is done the new image is returned
    """
    hl = int(len(im)/kernel_size)
    wl = int(len(im[0])/kernel_size)
    new_im = np.empty([hl,wl,3]) 
    for  y in range(hl):
        for x in range(wl):
            big = [0,0,0]
            for iy in range(kernel_size):
                for ix in range(kernel_size):
                    i=y*stride+iy
                    j=x*stride+ix
                    if(big[0]<im[i][j][0]):
                        big[0]=im[i][j][0]					
                    if(big[1]<im[i][j][1]):
                        big[1]=im[i][j][1]	
                    if(big[2]<im[i][j][2]):
                        big[2]=im[i][j][2]	
            new_im[y][x][:]=big
	

    return new_im
    ### END YOUR CODE HERE ### 


if __name__ == "__main__":

    # DO NOT CHANGE
    im = skimage.data.chelsea()
    im = utils.uint8_to_float(im)
    max_pooled_image = MaxPool2d(im, 4)

    utils.save_im("chelsea.png", im)
    utils.save_im("chelsea_maxpooled.png", max_pooled_image)

    im = utils.create_checkerboard()
    im = utils.uint8_to_float(im)
    utils.save_im("checkerboard.png", im)
    max_pooled_image = MaxPool2d(im, 2)
    utils.save_im("checkerboard_maxpooled.png", max_pooled_image)