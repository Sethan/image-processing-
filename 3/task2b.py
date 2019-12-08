import utils
import numpy as np
import skimage
from skimage import io

def region_growing(im: np.ndarray, seed_points: list, T: int) -> np.ndarray:
    """
        A region growing algorithm that segments an image into 1 or 0 (True or False).
        Finds candidate pixels with a Moore-neighborhood (8-connectedness). 
        Uses pixel intensity thresholding with the threshold T as the homogeneity criteria.
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
            seed_points: list of list containing seed points (row, col). Ex:
                [[row1, col1], [row2, col2], ...]
            T: integer value defining the threshold to used for the homogeneity criteria.
        return:
            (np.ndarray) of shape (H, W). dtype=np.bool
    """
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions
    """
    the image dimensions are stoed in xmax and ymax
    
    an list over active pixels is decleared
    
    for each seed in the seed point list 
    the seed is added to the activeList and 
    treshold values TN (T negative) and TP (T positive) are calculated by subtracting and adding the T value from the seed value
    
    as long as the active list is not empty the first element of the list is popped as "check"
    running trough the bordering 8 pixels by using a double four loop and making sure that y and x are not both zero
    if the value is in the image it is created as a neighbor tuple 
    if the tuple is within the treshold, not in the activeList and not allready True in segmented
    then its set to true and added to activeList
    
    
    """
    segmented = np.zeros_like(im).astype(bool)
    xmax=im.shape[0]
    ymax=im.shape[1]
    
    
    activeList=[]
   
    for seed in seed_points:
        activeList.append((seed[0],seed[1]))
        TN=im[seed[0]][seed[1]]-T
        TP=im[seed[0]][seed[1]]+T
        while activeList:
            check=activeList.pop(0)
            for y in range(-1,2):
                for x in range(-1,2):
                    if( not (x == 0 and  y == 0)):
                        if(x+check[0]>-1 and y+check[1]>-1 and x+check[0]<xmax and y+check[1]<ymax):
                            tuple = (check[0]+x,check[1]+y)
                            if((im[tuple[0]][tuple[1]]>TN) and (im[tuple[0]][tuple[1]]<TP) and segmented[tuple[0]][tuple[1]]!=True and (tuple not in activeList)):
                                segmented[tuple[0]][tuple[1]]= True
                                activeList.append(tuple)
        
    return segmented
    ### END YOUR CODE HERE ### 



if __name__ == "__main__":
    # DO NOT CHANGE
    im = utils.read_image("defective-weld.png")

    seed_points = [ # (row, column)
        [254, 138], # Seed point 1
        [253, 296], # Seed point 2
        [233, 436], # Seed point 3
        [232, 417], # Seed point 4
    ]
    intensity_threshold = 50
    segmented_image = region_growing(im, seed_points, intensity_threshold)

    assert im.shape == segmented_image.shape, \
        "Expected image shape ({}) to be same as thresholded image shape ({})".format(
            im.shape, segmented_image.shape)
    assert segmented_image.dtype == np.bool, \
        "Expected thresholded image dtype to be np.bool. Was: {}".format(
            segmented_image.dtype)

    segmented_image = utils.to_uint8(segmented_image)
    utils.save_im("defective-weld-segmented.png", segmented_image)

