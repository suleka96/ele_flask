from flask import Flask, request,jsonify,render_template,url_for,redirect
#import base64
import io
from shutil import copyfile

import os
import sys
import shutil
import cv2
from darkflow.net.build import TFNet
#import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
#import numpy as np
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications.xception import preprocess_input
from keras.models import load_model

import tensorflow as tf

print("inintlizing")

options = {
    'model': 'cfg/yolo2-obj.cfg',
    #'load': 'weights/yolo2-obj_8000.weights',
    'load': 'weight/yolo2-obj_corrected_5000.weights',
    'threshold': 0.3,
}

print("################################################################")
print("yolo model loading..")
tfnet = TFNet(options)
print("yolo model loaded")
print("################################################################")
print("xception loading")
model=load_model('xception_model/56_xception/weights.best_xception.hdf5')
print("xception loaded")
# this is key : save the graph after loading the model
global graph
graph = tf.get_default_graph()
print("################################################################")


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
global pred_ar
global image_names
global full_set
global user
user = None
full_set = os.listdir(os.path.join(APP_ROOT,"static/images/samples"))




app = Flask(__name__)
@app.route('/')
def index():
    global full_set
    global user
    if user is None:
        return redirect(url_for('login'))
    return render_template('index.html',full_set=full_set,user=user)

@app.route("/predicted")
def predicted():
    global pred_ar
    global full_set
    global image_names
    return render_template('upload_complete.html',image_names=image_names,pred_ar=pred_ar,full_set=full_set)


@app.route("/add_lbl", methods=['GET','POST'])
def add_lbl():
    global pred_ar
    global image_names
    new_label = request.form['new_label']
    img_name = request.form['img_name'].split("uniqueone")[1]
    target = os.path.join(APP_ROOT,"Labels/")+ new_label + "/"
    if not os.path.isdir(target):
        os.mkdir(target)
    dst = target + img_name

    src = os.path.join(APP_ROOT,"static/upload_images/")
    src = src + img_name
    copyfile(src, dst)
    pred_ar.remove(pred_ar[image_names.index(img_name)])
    image_names.remove(img_name)
    return redirect(url_for('predicted'))



@app.route('/anotate/<string:ele_name>/<string:img_name>', methods=['GET','POST'])
def anotate(ele_name,img_name):
    global pred_ar
    global full_set
    global image_names
    target = os.path.join(APP_ROOT,"Labels/")+ ele_name + "/"
    if not os.path.isdir(target):
        os.mkdir(target)

    dst = target + img_name

    src = os.path.join(APP_ROOT,"static/upload_images/")
    src = src + img_name

    #print("target: "+dst+"  "+ src)
    copyfile(src, dst)
    img_name = str(img_name)
    #print(image_names)
    #print(img_name)
    pred_ar.remove(pred_ar[image_names.index(img_name)])
    image_names.remove(img_name)
    return redirect(url_for('predicted'))


@app.route("/upload", methods=["POST"])
def upload():
    global pred_ar
    global image_names
    target = os.path.join(APP_ROOT,"static/upload_images/")
    if not os.path.isdir(target):
        os.mkdir(target)
    unlink_target(target)
    for upload in request.files.getlist("file"):
        filename = upload.filename
        destination = target+filename
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)
    pred_ar = predict(target)
    image_names = os.listdir(os.path.join(APP_ROOT,"static/upload_images/"))
    return redirect(url_for('predicted'))

def preprocess_image(in_image):
    #image = image.resize(target_size)
    out_image = image.img_to_array(in_image)
    #out_image = np.expand_dims(out_image,axis=0)
    return out_image

#@app.route("/predict")
def predict(target):
    folder = target
    save_folder = os.path.join(APP_ROOT,"static/outputs/")
    print(save_folder)
    unlink_target(save_folder)
    pred_ar =[]
    for img_name in (os.listdir(folder)):
        indi_ar = []
        path = folder+img_name
        img = cv2.imread(path)
        r = min(1000 / img.shape[0], 1000 / img.shape[1])
        w  = int(r * img.shape[0])
        h = int(r * img.shape[1])
        img = cv2.resize(img, (h,w))
        save_path_img = save_folder +img_name
        save_path_prdiction = save_folder +img_name[:-4]+".txt"
        #print(save_path_prdiction)

        result = tfnet.return_predict(img)
        total = 0
        f = open(save_path_prdiction,"w")
        f.write(str(len(result))+'\n')
        indi_ar.append(str(len(result)))
        for i in range(len(result)):
            total = total+1
            x1 = result[i]['topleft']['x']
            y1 = result[i]['topleft']['y']
            x2 = result[i]['bottomright']['x']
            y2 = result[i]['bottomright']['y']
            label = result[i]['label']
            #print(label)
            # add the box and label and display it
            #img = cv2.rectangle(img, tl, br, (0, 255, 0), 20)
            crop_ear = img[y1:y2, x1:x2]
            #img2 = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            #plt.imshow(crop_img)
            #plt.show()
            label = result[i]['label'] + str(i+1)
            # add the box and label and display it
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 7)
            img = cv2.putText(img, label, (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

            cv2.imwrite(save_path_img,img)
            crop_ear = cv2.resize(crop_ear,(299,299),interpolation=cv2.INTER_AREA)
            crop_ear = Image.fromarray(crop_ear)

            crop_ear = image.img_to_array(crop_ear)
            #print(x.shape)
            crop_ear = np.expand_dims(crop_ear, axis=0)
            #print(x.shape)
            crop_ear = preprocess_input(crop_ear)
            with graph.as_default():
    	           preds = model.predict(crop_ear)
            label_lines = [line.rstrip() for line in tf.gfile.GFile("xception_model/56_xception/lbl.txt")]

            #txt_str = "prediction: "+label_lines[preds.argmax(axis=-1)[0]]+" : "+ str(int((preds[0][preds.argmax(axis=-1)[0]])*100))+"%"
            #print(txt_str)
            #print("1")
            t = 0
            print("2")
            while(t<5):
                t = t +1
                print("3")
                f.write(label_lines[preds.argmax(axis=-1)[0]]+'\n')
                indi_ar.append(label_lines[preds.argmax(axis=-1)[0]])
                val = "%.2f" % ((preds[0][preds.argmax(axis=-1)[0]])*100)
                f.write(str(val)+'\n')
                indi_ar.append(str(val))
                print(label_lines[preds.argmax(axis=-1)[0]])
                print(val)
                preds[0][preds.argmax(axis=1)[0]] = 0
                #print(preds.argmax(axis=1)[0])
        pred_ar.append(indi_ar)

    f.close()
    print("done")
    return pred_ar

def unlink_target(target):
    for the_file in os.listdir(target):
        file_path = target+the_file
        #print("xxx: "+file_path)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print("unlink_done")
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


@app.route('/sample')
def running():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            global full_set
            global user
            user = request.form['username']
            return redirect(url_for('index'))
    return render_template('login.html', error=error)

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    target = os.path.join(APP_ROOT,"static/upload_images/")
    unlink_target(target)
    global user
    global pred_ar
    global image_names
    user = None
    pred_ar = None
    image_names = None
    return redirect(url_for('index'))
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT',80))
    app.run(host='0.0.0.0',port=port)
