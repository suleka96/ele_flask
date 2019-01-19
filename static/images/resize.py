import cv2
import glob
c = 0

for filename in glob.glob('samples/*.JPG'): #assuming gif
    c = c+1
    print(filename+" "+str(c))
    image =  cv2.imread(filename)
    resized_image = cv2.resize(image, (450, 350))
    cv2.imwrite(filename, resized_image)
