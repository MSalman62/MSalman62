from flask import Flask, render_template, request, send_file
import numpy as np
#import os
#print("Current Working Directory:", os.getcwd())

cats = {0: 'Bacterial Pustule',
 1: 'Frogeye Leaf Spot',
 2: 'Healthy',
 3: 'Rust',
 4: 'Sudden Death Syndrome',
 5: 'Target Leaf Spot',
 6: 'Yellow Mosaic'}

from PIL import Image
from keras.models import load_model
app = Flask(__name__, static_folder='static', static_url_path='/static/')

modal = load_model('model and data/SLDP.h5')

def predict_label(img_path):
	image = Image.open(img_path)
	image = image.resize((224, 224))
	img = np.array(image)
	img_batch = np.expand_dims(img, axis=0)
	img = img/255
	pred_index = np.argmax(modal.predict(img_batch))
	pred_class=cats[pred_index]
	return "\n{}".format(pred_class)

# Flask Code

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("project.html")

@app.route("/about/", methods = ['GET'])
def about_page():
	return """Hello there!!!"""
import os
# @app.route("/submit/", methods = ['GET', 'POST'])
@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        if img:
            img_path = os.path.join("static", img.filename)
            img.save(img_path)
            pred_disease = predict_label(img_path)
            return render_template("project.html", predicted_class=pred_disease, img_path=img_path)
        else:
            return render_template("project.html", predicted_class="No image uploaded")
    return render_template("project.html")


@app.route("/TestImages/<fname>", methods = ['GET'])
def get_img(fname):
    return send_file(f'TestImages/{fname}')

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
      
