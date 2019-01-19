import os
import sys
print(sys.argv[1])
os.makedirs("This_is_folder")

import cv2
#from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import numpy as np
from keras.preprocessing import image
#from keras.applications.imagenet_utils import preprocess_input
from keras.applications import imagenet_utils
#from imagenet_utils import decode_predictions
from keras.applications.xception import preprocess_input
from keras.models import load_model
import tensorflow as tf


print("yolo model loading.")
options = {
    'model': '../darkflow/cfg/yolo2-obj.cfg',
    #'load': 'weights/yolo2-obj_8000.weights',
    'load': '../darkflow/weight/yolo2-obj_corrected_5000.weights',
    'threshold': 0.3,
}
print("yolo model loading..")
tfnet = TFNet(options)
print("yolo model loaded")
'''
model=load_model('xception_model/56_xception/weights.best_xception.hdf5')
print("xception loaded")

folder = "Images_56test_data"
#folder2 = "Evaluation_Images"
pad = 0
total = 0
correct_count = 0
top5_correct_count = 0
for ele_name in (os.listdir(folder)):
    for imgname in (os.listdir(folder+"/"+ele_name)):
        img = cv2.imread(folder+"/"+ele_name+"/"+imgname)
        r = min(1000 / img.shape[0], 1000 / img.shape[1])
        w  = int(r * img.shape[0])
        h = int(r * img.shape[1])
        img = cv2.resize(img, (h,w))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = img.copy()
        #img1 = cv2.imread(folder+"/"+ele_name+"/"+imgname)
        #img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = tfnet.return_predict(img)

        for i in range(len(result)):
            total = total+1
            x1 = result[i]['topleft']['x']-pad
            y1 = result[i]['topleft']['y']-pad
            x2 = result[i]['bottomright']['x']+pad
            y2 = result[i]['bottomright']['y']+pad
            label = result[i]['label']
            # add the box and label and display it
            #img = cv2.rectangle(img, tl, br, (0, 255, 0), 20)

            crop_ear = img[y1:y2, x1:x2]
            #img2 = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            #plt.imshow(crop_img)
            #plt.show()
            crop_ear = cv2.resize(crop_ear,(299,299),interpolation=cv2.INTER_AREA)
            crop_ear = Image.fromarray(crop_ear)

            crop_ear = image.img_to_array(crop_ear)
            #print(x.shape)
            crop_ear = np.expand_dims(crop_ear, axis=0)
            #print(x.shape)
            crop_ear = preprocess_input(crop_ear)
            preds = model.predict(crop_ear)
            label_lines = [line.rstrip() for line in tf.gfile.GFile("xception_model/56_xception/lbl.txt")]


            txt_str = "prediction: "+label_lines[preds.argmax(axis=-1)[0]]+" : "+ str(int((preds[0][preds.argmax(axis=-1)[0]])*100))+"%"
            print(txt_str)
            if(ele_name==label_lines[preds.argmax(axis=-1)[0]]):
                correct_count = correct_count+1
                top5_correct_count = top5_correct_count+1
                print("correct ="+str(top5_correct_count)+str(correct_count))
            else:
                t = 0
                while(t<5):
                    t = t +1
                    preds[0][preds.argmax(axis=1)[0]] = 0
                    #print(preds.argmax(axis=1)[0])
                    if(ele_name==label_lines[preds.argmax(axis=-1)[0]]):
                        top5_correct_count = top5_correct_count+1
                        print("top5 correct = "+str(top5_correct_count))
                        break




print("total = "+str(total))
print("correct = "+str(correct_count))
print("secitivity = "+str(correct_count/total*100)+"%")

'''
