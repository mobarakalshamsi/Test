from sklearn import datasets
from joblib import load
import numpy as np
import json

#load the model

#my_model = load('svc_model.pkl')

class_names = ["alive", "dead"]

def my_prediction(id):
    my_model = load('svc_model.pkl')
    dummy = np.array(id)
    dummyT = dummy.reshape(1,-1)
    #dummy_str = dummy.tolist()  	
    #r = dummy.shape
    #t = dummyT.shape
    #r_str = json.dumps(r)
    #t_str = json.dumps(t)
    prediction = my_model.predict(dummyT)
    prediction = int(prediction)
    name = class_names[prediction]
    #name = name.tolist()
    #name_str = json.dumps(name)
    #pred_str = prediction.tolist()
    #pred_str = json.dumps(pred_str)
    #dummy_str = json.dumps(dummy_str)
    #class_str = class_names.tolist()
    #class_str = json.dumps(class_str)
    str = name
    return str
