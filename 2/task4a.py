import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils
from skimage import draw



def convolve_im(im: np.array,
                fft_kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the frequency kernel (fft_kernel),
        and returns the resulting image.

        "verbose" can be used for visualizing different parts of the 
        convolution

    Args:
        im: np.array of shape [H, W]
        fft_kernel: np.array of shape [H, W] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    kernel=fft_kernel
    conv_result = im    
    conv_result = skimage.morphology.closing(conv_result,kernel)
    """
	first it applies fourier transforms on the the image
	then it sets the image to be the pointwise multiplication of the transforms

    the image is inverse fourier transformed and filtered for real values
    
    the fourier transform of the image, kernel and sum are also shifted and filtered for set to be their absolute values
	lastly everything is displayed in the subplots
    """
    if verbose:
        
        fftImage=np.fft.fft2(conv_result)
		
		
		
        fftImageTransformed=np.multiply(fftImage,kernel)
		
        
        conv_result=np.fft.ifft2(fftImageTransformed)
        
        conv_result=np.real(conv_result)

        fftImageTransformed=np.fft.fftshift(fftImageTransformed)
        fftImage=np.fft.fftshift(fftImage)
        kernel=np.fft.fftshift(kernel)

        fftImageTransformed=np.absolute(fftImageTransformed)
        fftImage=np.absolute(fftImage)
        kernel=np.absolute(kernel)
		
		
        # Use plt.subplot to place two or more images beside eachother
        plt.figure(figsize=(20, 4))
        # plt.subplot(num_rows, num_cols, position (1-indexed))
        plt.subplot(1, 5, 1)
        plt.imshow(im, cmap="gray")
        plt.subplot(1, 5, 2)
        plt.imshow(fftImage, cmap="gray")
        plt.subplot(1, 5, 3)
        plt.imshow(kernel, cmap="gray")
        plt.subplot(1, 5, 4)
        plt.imshow(fftImageTransformed, cmap="gray")
        plt.subplot(1, 5, 5)
        plt.imshow(conv_result, cmap="gray")
    ### END YOUR CODE HERE ###
    return conv_result


if __name__ == "__main__":
    verbose = True

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)
    # DO NOT CHANGE
    frequency_kernel_low_pass = utils.create_low_pass_frequency_kernel(im, radius=50)
    image_low_pass = convolve_im(im, frequency_kernel_low_pass,
                                 verbose=verbose)
    # DO NOT CHANGE
    frequency_kernel_high_pass = utils.create_high_pass_frequency_kernel(im, radius=50)
    image_high_pass = convolve_im(im, frequency_kernel_high_pass,
                                  verbose=verbose)

    if verbose:
        plt.show()
    utils.save_im("camera_low_pass.png", image_low_pass)
    utils.save_im("camera_high_pass.png", image_high_pass)
