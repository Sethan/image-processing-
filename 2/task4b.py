import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils




def convolve_im(im: np.array,
                kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the spatial kernel (kernel),
        and returns the resulting image.

        "verbose" can be used for visualizing different parts of the 
        convolution.
        
        Note: kernel can be of different shape than im.

    Args:
        im: np.array of shape [H, W]
        kernel: np.array of shape [K, K] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    """
	compared to the 4a solution this just adds padding to the filter if its smaller than the image
	this is done by using the second parameter in fft.fft2 
	
	first it applies fourier transforms on the kernel and the image
	then it sets the image to be the pointwise multiplication of the transforms

    the image is inverse fourier transformed and filtered for real values
    the domain image is shifted and taken the absolute value of
    the fourier transform of the image and kernel are also shifted and set to be the absolute value
	lastly everything is displayed in the subplots
    """
    conv_result = im    
 
    if verbose:
        fftKernel=np.fft.fft2(kernel,im.shape)
        fftImage=np.fft.fft2(conv_result)
		
		
		
        conv_result=np.multiply(fftImage,fftKernel)
        fftImageTransformed=conv_result
		
        
        conv_result=np.fft.ifft2(conv_result)
        
        conv_result=np.real(conv_result)

        fftImageTransformed=np.fft.fftshift(fftImageTransformed)
        fftImage=np.fft.fftshift(fftImage)
        fftKernel=np.fft.fftshift(fftKernel)

        fftImageTransformed=np.absolute(fftImageTransformed)
        fftImage=np.absolute(fftImage)
        fftKernel=np.absolute(fftKernel)
		
		
        # Use plt.subplot to place two or more images beside eachother
        plt.figure(figsize=(20, 4))
        # plt.subplot(num_rows, num_cols, position (1-indexed))
        plt.subplot(1, 5, 1)
        plt.imshow(im, cmap="gray")
        plt.subplot(1, 5, 2)
        plt.imshow(fftImage, cmap="gray")
        plt.subplot(1, 5, 3)
        plt.imshow(fftKernel, cmap="gray")
        plt.subplot(1, 5, 4)
        plt.imshow(fftImageTransformed, cmap="gray")
        plt.subplot(1, 5, 5)
        plt.imshow(conv_result, cmap="gray")
    ### END YOUR CODE HERE ###
    return conv_result


if __name__ == "__main__":
    verbose = True  # change if you want

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)

    # DO NOT CHANGE
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    image_gaussian = convolve_im(im, gaussian_kernel, verbose)

    # DO NOT CHANGE
    sobel_horizontal = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    image_sobelx = convolve_im(im, sobel_horizontal, verbose)

    if verbose:
        plt.show()

    utils.save_im("camera_gaussian.png", image_gaussian)
    utils.save_im("camera_sobelx.png", image_sobelx)
