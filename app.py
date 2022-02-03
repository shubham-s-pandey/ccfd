from flask import Flask, render_template, url_for, request
import pandas as pd, numpy as np
import pickle
import werkzeug
# load the model from disk
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('cc.html')


@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		me = request.form['message']
		message = [float(x) for x in me.split()]
		vect = np.array(message).reshape(1, -1)
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)


@app.route("/simulate404")
def simulate404():
    abort(404)
    return render_template("html.html")

@app.route("/simulate500")
def simulate500():
    abort(500)
    return render_template("html.html")
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/cc")
def cc():
    return render_template("home.html")
@app.route("/datasets")
def datasets():
    return render_template("csv.html")
@app.errorhandler(404)
def not_found_error(error):
    return render_template('html.html'), 404

@app.errorhandler(werkzeug.exceptions.HTTPException)
def internal_error(error):
    return render_template('html.html'), 500



if __name__ == '__main__':
	app.run(debug=False)

	
