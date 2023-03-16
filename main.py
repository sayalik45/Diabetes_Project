from flask import Flask,request,render_template
from utils import diabetes_prediction

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data',methods = ['POST','GET'])
def get_data():
    data = request.form
    class_obj = diabetes_prediction(data)
    result = class_obj.diab_predict()

    return render_template('index.html',prediction = result)

if __name__ == "__main__":
    app.run(host = '0.0.0.0')