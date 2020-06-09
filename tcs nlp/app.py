# -*- coding: utf-8 -*-
from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.externals import joblib
import hotel_review_script
hr1 = hotel_review_script.hotel_review()
model = open('model.pkl','rb')
lr1 = joblib.load(model)

app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    d = pd.read_csv("hotel_review_final.csv")
    column_values = d["Hotel_Name"]. values. ravel()
    li1 = pd. unique(column_values)
    return render_template('home.html', li1 = li1)

@app.route('/predict',methods=['POST'])

def predict():
    data=pd.read_csv("hotel_review_final.csv")
    data_review=data.copy()
    X_train, X_test, y_train, y_test = hr1.splitting(data_review)
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        X=hr1.input_pipeline(data)
        my_prediction = lr1.predict(X)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)