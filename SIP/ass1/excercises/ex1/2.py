from matplotlib.pyplot import axis, imshow, show 
from skimage.io import imread
# Read in intensity image
A = imread('cameraman.png');
# Display image using imshow - but the colors are wrong!
imshow(A) # Prepare to show the image in A
show() # This opens a window for each imshow call and displays the content
# Remove axes including tick marks and labels
axis('on')
# Display intensity image in gray-scale
imshow(A, cmap='gray') # Tell matplotlib to interpret A as a gray scale image show()
