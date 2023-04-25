import numpy as np
nombres=['Nada','Piso','Paredes','Objetos','Objetos pequeños','Muebles','Electrónica','Cocina']
precision_por_imagen=[]
for i in range(0,145,1):
  y_pred=result_mask[i]
  y_test=test_mask[i]
  aciertos = []
  for valor in range(8):
    total = 0
    correctos = 0
    for i in range(128):
      for j in range(128):
        if y_test[i, j] == valor:
          total += 1
          if y_pred[i, j] == valor:
            correctos += 1
    if total > 0:
      aciertos.append(correctos / total)
    else:
      aciertos.append(0)
  precision_por_imagen.append(aciertos)
precision_por_imagen = np.array(precision_por_imagen)
print(np.mean(precision_por_imagen))
plt.imshow(result_mask[10])
plt.show()
plt.imshow(test_mask[10])
plt.show()

# Creamos el DataFrame a partir de los datos y los nombres de columna
df = pd.DataFrame(data=precision_por_imagen, columns=nombres)

# Agregamos una columna con el número de imagen (0 a 144)
df.insert(0, "N° de imagen", range(145))

# Imprimimos el DataFrame resultante
display(df)

# Inicializar la lista de IoUs para cada clase

iou_imagenes=[]
for i in range(0,145,1):
  # Iterar sobre cada clase de 0 a 7
  array1=result_mask[i]
  array2=test_mask[i]
  iou_per_class = []
  for c in range(8):
    # Convertir los arrays en máscaras binarias para la clase correspondiente
    mask1 = (array1 == c).astype(int)
    mask2 = (array2 == c).astype(int)

    # Calcular la intersección entre las máscaras binarias
    intersection = np.sum(np.logical_and(mask1, mask2))

    # Calcular la unión entre las máscaras binarias
    union = np.sum(mask1) + np.sum(mask2) - intersection

    # Calcular el IoU para la clase correspondiente
    iou = intersection / union

    # Agregar el IoU a la lista de IoUs por clase
    iou_per_class.append(iou)
  iou_imagenes.append(iou_per_class)
# Imprimir los IoUs para cada clase
iou_imagenes = np.array(iou_imagenes)
print(iou_imagenes.shape)
print(np.mean(iou))
