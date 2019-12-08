import numpy as np
import skimage
from skimage import io
import utils
import pathlib


def otsu_thresholding(im: np.ndarray) -> int:
    """
        Otsu's thresholding algorithm that segments an image into 1 or 0 (True or False)
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
        return:
            (int) the computed thresholding value
    """
    assert im.dtype == np.uint8
    ### START YOUR CODE HERE ### (You can change anything inside this block) 
    # You can also define other helper functions
    # Compute normalized histogram
    
    #find histogram
    
    """
    First the a two 1d arrays of sizes the same as the image and 256 are created
    and for each occurance of a pixel value x the histograms x index value is incremented
    a normalized version is decleared where histogram values are divided by the sizes
    
    mG is the cumelative average 
    
    two values to find the best threshold are decleared
    and for each value in the range 0-255 a treshold is calculated using the formulas
    
    a treshold with either P1 or P2 = 0 is ignored
    
    if the betweenClass value is larger than the previous the treshold is updated
    """
    im1D=im.flatten()
    MN=im.shape[0]*im.shape[1]  
    histogram=np.zeros(256)
    for i in im1D:
        histogram[i] += 1
    histogramNormalized=np.zeros(256)
    
    
    mG=0
    for n in range(len(histogram)):
        histogramNormalized[n]=histogram[n]/MN
        mG+=histogramNormalized[n]*histogram[n]
    
    thresholdBest = 0
    oldMax=-99999999
    for threshold in range(256):    
        mK1=0
        mK2=0
        P1=0
        for n in range(threshold):
            mK1+=histogramNormalized[n]*histogram[n]
            P1+=histogramNormalized[n]
        for n in range(threshold,256):
            mK2+=histogramNormalized[n]*histogram[n]
        
        
        P2=1-P1        
        if(P1!=0 and P2!=0):
            m1=mK1/P1
            m2=mK2/P2

            betweenClass=P1*np.power((m1-mG),2)+P2*np.power((m2-mG),2)
            globalVar=0
            for n in range(len(histogramNormalized)):
                globalVar+=histogramNormalized[n]*np.power((n-mG),2)
                
            
            normalized=betweenClass/globalVar
            if(betweenClass>oldMax):
                thresholdBest=threshold
                oldMax=betweenClass

    return thresholdBest
    ### END YOUR CODE HERE ### 


if __name__ == "__main__":
    # DO NOT CHANGE
    impaths_to_segment = [
        pathlib.Path("thumbprint.png"),
        pathlib.Path("polymercell.png")
    ]
    for impath in impaths_to_segment:
        im = utils.read_image(impath)
        threshold = otsu_thresholding(im)
        print("Found optimal threshold:", threshold)

        # Segment the image by threshold
        segmented_image = (im >= threshold)
        assert im.shape == segmented_image.shape, \
            "Expected image shape ({}) to be same as thresholded image shape ({})".format(
                im.shape, segmented_image.shape)
        assert segmented_image.dtype == np.bool, \
            "Expected thresholded image dtype to be np.bool. Was: {}".format(
                segmented_image.dtype)

        segmented_image = utils.to_uint8(segmented_image)

        save_path = "{}-segmented.png".format(impath.stem)
        utils.save_im(save_path, segmented_image)


