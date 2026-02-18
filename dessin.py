from PIL import Image
import numpy as np
image = Image.open("test.png")
print(image.height, image.width)

table = np.array(image.getdata()).reshape(image.height, image.width, 4)
table = table[:,:,0:3]
print(np.min(table), np.max(table))
rouges = [(i,j) 
          for i in range(image.height)      
          for j in range(image.width) 
          if table[i,j,0]>=250 and table[i,j,1]<=10
          and table[i,j,2]<=10]

print(len(rouges))
