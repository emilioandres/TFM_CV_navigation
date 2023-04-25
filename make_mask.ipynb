cats = { 'nada': [1],
 'floor': [2],
 'wall': [3],
 'objects': [4],
 'in_floor': [5],
 'furniture': [6],
 'electronics': [7],
 'kitchen': [8]}
def make_mask(img):
    mask = np.zeros((img.shape[0], img.shape[1], 8))
    for i in range(1, 9):
        if i in cats['nada']:
            mask[:,:,0] = np.logical_or(mask[:,:,0],(img==i))
        elif i in cats['floor']:
            mask[:,:,1] = np.logical_or(mask[:,:,1],(img==i))
        elif i in cats['wall']:
            mask[:,:,2] = np.logical_or(mask[:,:,2],(img==i))
        elif i in cats['objects']:
            mask[:,:,3] = np.logical_or(mask[:,:,3],(img==i))
        elif i in cats['in_floor']:
            mask[:,:,4] = np.logical_or(mask[:,:,4],(img==i))
        elif i in cats['furniture']:
            mask[:,:,5] = np.logical_or(mask[:,:,5],(img==i))
        elif i in cats['electronics']:
            mask[:,:,6] = np.logical_or(mask[:,:,6],(img==i))
        elif i in cats['kitchen']:
            mask[:,:,7] = np.logical_or(mask[:,:,7],(img==i))
    mask = np.resize(mask,(img.shape[0]*img.shape[1], 8))
    return mask
    
Y_test=[]
X_mask=[]
for i in segm_result:
  X_mask.append(make_mask(i))
for i in Y_mask_test:
  Y_test.append(make_mask(i))
X_mask=np.array(X_mask)
X_mask = X_mask.astype('uint8')
X_mask=np.array(X_mask)
dataset_img=dataset_img.astype('uint8')
dataset_img=np.array(dataset_img)
depth_result =depth_result.astype('uint8')
depth_result=np.array(depth_result)
Y_test = np.array(Y_test)
print(X_mask.shape)
print(Y_test.shape)
