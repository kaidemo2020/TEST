import numpy as np
from flask import Flask, request, jsonify, render_template, make_response
from functools import wraps

import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open('model_3m_svc1.pkl', 'rb'))



def auth_required(f):
    @wraps(f)
    def decorated(*args,**kwargs):
        auth = request.authorization
        if auth and auth.username == 'usr1' and auth.password == '123':
            return f(*args, **kwargs)
        
        return make_response('could not varify',401,{'WWW-Authenticate' : 'Basic realm = "Login Required"'})

    return decorated   



@app.route('/')
@auth_required
def home():
    return render_template('index2.html',prediction_text='Welcome')



@app.route('/data', methods=['GET', 'POST'])
@auth_required
def data():
    
    
    if request.method == 'POST':
        
        df = pd.read_excel(request.files.get('file'))
        
        
        data = pd.read_excel(request.files.get('file'))
       
        df= df.apply(lambda x: x.astype(str).str.lower())
        result = model.predict(df)
        prob = model.predict_proba(df)
        result = pd.to_numeric(result)
        #result = result.astype(int)
        Maxprob = prob.max(axis=1)*100
        Maxprob = np.around(Maxprob, 2)
        #data= data.apply(lambda x: x.astype(str).str.upper())
        data['Pnumber'] = result.tolist()
        #data['Number'] = data['Number'].astype(int)
        data['Probability'] = Maxprob.tolist()
        #data= data.apply(lambda x: x.astype(int))
        #data.to_excel(r'C:\Users\LENOVO\Desktop\Project\result.xlsx', index = False)
        
        
        return render_template('data.html', data=data.to_dict() )


@app.route('/predict',methods=['POST'])
@auth_required
def predict():
    
    
    int_features = [[x for x in request.form.values()]]
    
    df = pd.DataFrame(int_features, columns=['Description','Country of Origin','Text Division','Profit Center for Semi/Finished products','Product Hierarchy','Sales Code'])
    
    df= df.apply(lambda x: x.astype(str).str.lower())
    
    prediction = model.predict(df)
    prob = model.predict_proba(df)
    probability = prob.max()*100
    probability = np.around(probability, 2)
    return render_template('index2.html', prediction_text='Predicted HTS Code {}  with {} % Confidence'.format(prediction,probability))

@app.route('/data1',methods=['POST'])
@auth_required
def data1():
    
    if request.method == 'POST':
        df = pd.read_excel(request.files.get('file'))
        
        
        data = pd.read_excel(request.files.get('file'))
       
        df= df.apply(lambda x: x.astype(str).str.lower())
        result = model.predict(df)
        prob = model.predict_proba(df)

        Maxprob = prob.max(axis=1)*100
        Maxprob = np.around(Maxprob, 2)
        data= data.apply(lambda x: x.astype(str).str.upper())
        data['Number'] = result.tolist()
        data['Probability'] = Maxprob.tolist()
        #data= data.apply(lambda x: x.astype(str).str.upper())
        #data.to_excel(r'C:\Users\LENOVO\Desktop\Project\result.xlsx', index = False)
        
        
        return render_template('data1.html', data=data.to_dict() )

if __name__ == "__main__":
    app.run(debug=True)