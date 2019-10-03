# analysis of vocal cord videos by deep networks of neurons
For this internship I will be in charge of setting up a deep learning network of neurons able to provide significant descriptors of vocal cords to detect the presence or absence of pathologies
For that we choose the Unet model as our application to segment and extract the glottal area region:

# UNet model 
=============================================================================================================================================
The architecture looks like a ‘U’ which justifies its name. This architecture consists of three sections: 
•	The contraction. 
•	The bottleneck. 
•	The expansion section.
The contraction section is made of many contraction blocks: Each block takes an input applies two 3X3 convolution layers followed by a 2X2 max pooling. The number of kernels or feature maps after each block doubles so that architecture can learn the complex structures effectively. 
the keras methods used in the model are:
- Conv2D: This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
Arguments
filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions.
padding: one of "valid" or "same" 

- Maxpooling2D: Max pooling is a sample-based discretization process. The objective is to down-sample an input representation (image, hidden-layer output matrix, etc.), reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned.
(see the jupyter note book for the implementation)


=============================================================================================================================================
# Testing our model:
In our application we have to make a binary segmentation between the foreground (the vocal folds area) and the remaining part of image which is the background and we get as an output a segmented image of the part that we want to delimitate.

In the code (that is found in the jupyter notebook) we invit the user to enter his image or set of images (a sequence) and apply our trained model on the images after rescaling them to adapt them to the input layer of the model. Then we generate a prediction of each image (the mask), plotting the image, the mask generated and the combined image showing the segmentation result.
In the end we calculate the glottal area GAW based on the binary mask generated which contain non zero values in the region of vocal folds for each image and in the end of the sequence we plot the GAW figure as it twill be shown in the next page.

In the end there's a binary classifier (SVM in our case ,arbitrary choice of the model) will indicate based on the glottal area sequence of values a probable pathology existing or not.


During the internship I have the opportunity to develop a Unet model that predict binary mask of images to delimitate the region of vocal folds and every thing is explained in the jupyter notebook file.
