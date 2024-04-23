# Author: Jan Niklas Kolf, 2020
from face_image_quality import SER_FIQ
import cv2
import numpy as np

if __name__ == "__main__":
    # Sample code of calculating the score of an image
    
    # Create the SER-FIQ Model
    # Choose the GPU, default is 0.
    ser_fiq = SER_FIQ(gpu=0)
        
    # Load the test image
    test_img = cv2.imread("./data/test_img.jpeg")
    
    # Align the image
#     aligned_img = ser_fiq.apply_mtcnn(test_img)
    
    # Calculate the quality score of the image
    # T=100 (default) is a good choice
    # Alpha and r parameters can be used to scale your
    # score distribution.
    
#     print('image.shape===========', image.shape)
    if test_img.shape[0]!=112 or test_img.shape[1]!=112:
#         print('cv2.resize')
        test_img = cv2.resize(test_img, (112, 112))
    image = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2,0,1))
    
    score = ser_fiq.get_score(image, T=100)
    
    print("SER-FIQ quality score of image 1 is", score)
    
    # Do the same thing for the second image as well
    test_img2 = cv2.imread("./data/test_img2.jpeg")
    
#     aligned_img2 = ser_fiq.apply_mtcnn(test_img2)
    
    score2 = ser_fiq.get_score(test_img2, T=100)
   
    print("SER-FIQ quality score of image 2 is", score2)
