def motion(arr_path, save_path):
  plt.figure()
  number = len(arr_path)
  for i in range(number-1):
    img1 = imread(arr_path[i])
    img2 = imread(arr_path[i+1])
    img = img_as_uint(np.abs(img1.astype('int16') - img2))
    plt.subplot(3,3,i+1)
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
  plt.show()
    
