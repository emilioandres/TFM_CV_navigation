color_map = {
'0': [0, 0, 0],
 '1': [255, 0, 0],
 '2': [0, 0, 255],
 '3': [255, 0, 127],
 '4': [0, 255, 0],
 '5': [0, 204, 204],
 '6': [153, 153, 0],
 '7': [255, 204, 204],
 '8': [26,120,97]
}

def pp(image,depth):
  alpha = 0.2
  dims = image.shape
  z = p_unet.predict((np.expand_dims(image, axis=0),np.expand_dims(depth, axis=0)))
  z = np.squeeze(z)
  z = z.reshape(128,128, 8)
  z = cv2.resize(z, (dims[1], dims[0]))
  y = np.argmax(z, axis=2)
  img_color = image.copy()   
  for i in range(dims[0]):
      for j in range(dims[1]):
          img_color[i, j] = color_map[str(y[i, j])]
  img_color=cv2.addWeighted(image, alpha, img_color, 1-alpha, 0, img_color)
  return y
list_random = [random.uniform(0, 100) for i in range(7)]
for i in list_random:
  i = int(i)
  z=Y_test[i].reshape(128,128, 8)
  z = cv2.resize(z, (128, 128))
  y = np.argmax(z, axis=2)
  img=pp(X_img_test[i],X_depth_test[i])
  fig, axs = plt.subplots(1, 4)
  fig.set_size_inches(15,15)
  axs[0].imshow(X_img_test[i])
  axs[1].imshow(X_depth_test[i])
  axs[2].imshow(img)
  axs[3].imshow(y)
