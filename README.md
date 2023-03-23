# Gender Recognition CNN

Paper: Dhomne, Amit, Ranjit Kumar, and Vijay Bhan. "**Gender recognition through face using deep learning**." _Procedia computer science_ 132 (2018): 2-10.

## Task
Gender Recognition was started with the problem in psychophysical studies to classify gender from human face; it concentrates on the e orts of perceiving human visual processing and recognizing relevant features that can be used to distinguish between female and male individuals. Exploration has proved that the discrepancy between a female face and male face can be used effectively to improvised the result of face recognition software in bio-metrics devices

Human gender detection which is a part of facial recognition has received extensive attention because of it’s different kind of application. Previous research works on gender detection have been accomplished based on different static body feature for example face, eyebrow, hand-shape, body-shape, finger nail etc. In this research work, we have presented human gender classification using Convolution Neural Network (CNN) from human face images as CNN has been recognised as best algorithm in the field of image classification.


## Database
CelabA - reshape 128 x128x3
**CelebFaces Attributes Dataset (CelebA)** is a large-scale face attributes dataset with more than **200K** celebrity images, each with **40** attribute annotations. The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations, including

-   **10,177** number of **identities**,
    
-   **202,599** number of **face images**, and
    
-   **5 landmark locations**, **40 binary attributes** annotations per image.


For this task specifically 1699 males and 1699 females where choosen for training and and another 100 males and 100 females for testing.

## Network

![image](https://user-images.githubusercontent.com/86794414/227157559-ae3810c8-e257-429f-811d-2a91b0fe4979.png)
