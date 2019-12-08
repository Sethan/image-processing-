import utils
import skimage
import skimage.morphology
import numpy as np
from task3a import remove_noise

"""
an zero array of the image size is decleared 
and going trough 255 iterations 
the image is eroded 
if the image[x][y] coordinate is true then results[x][y] is increased with 1
 

"""
def erode(im: np.ndarray, selem: np.ndarray) -> np.ndarray:
    result=np.zeros(im.shape)
    for counter in range(255):
        im=skimage.morphology.binary_erosion(im,selem)
        for x in range(im.shape[0]):
            for y in range(im.shape[1]):
                if(im[x][y]):
                    result[x][y]+=1
        
        
    
    return result


def distance_transform(im: np.ndarray) -> np.ndarray:
    """
        A function that computes the distance to the closest boundary pixel.

        args:
            im: np.ndarray of shape (H, W) with boolean values (dtype=np.bool)
        return:
            (np.ndarray) of shape (H, W). dtype=np.int32
    """
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions
    structuring_element = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=bool)

    result=erode(im,structuring_element).astype(np.int32)
    

    return result
    ### END YOUR CODE HERE ### 



if __name__ == "__main__":
    im = utils.read_image("noisy.png")
    binary_image = (im != 0)
    noise_free_image = remove_noise(binary_image)
    distance = distance_transform(noise_free_image)

    assert im.shape == distance.shape, \
        "Expected image shape ({}) to be same as resulting image shape ({})".format(
            im.shape, distance.shape)
    assert distance.dtype == np.int32, \
        "Expected resulting image dtype to be np.int32. Was: {}".format(
            distance.dtype)

    distance = utils.to_uint8(distance)
    utils.save_im("noisy-distance.png", distance)

    
    



