from skimage.io import imread, imsave
photo = 'cameraman.png'
A = imread(photo) # Read in an image
print(type(A)) # The type of an image is a numpy.ndarray
print(A.shape) # Print the dimensions (in pixels) of the image 
print(A.dtype) # The data type of each pixel (element in the numpy.ndarray) 
imsave('cameraman.jpg', A) # Write an image
