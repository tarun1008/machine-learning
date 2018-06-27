import pandas as pd
import json
import dill as pickle
from flask import Flask, jsonify, request, Response

app = Flask(__name__)

@app.route('/predict', methods=['POST', 'GET'])
def apicall():
        returnJson = {"date": "08-12-2016", "customername": "Lane Constructions", "device": "DRILL", "component": ["Power Unit", "Drill Shaft", "Motor", "Clutch", "Quill Assembly", "Angle Plates"], "category": "Power Tools", "prediction": [0, 0, 1, 0, 1, 1]}
        resp = Response(response=json.dumps(returnJson), status=200, mimetype="application/json")
        return(resp)
        
if __name__ == '__main__':
    app.run(host='localhost', debug=False, use_reloader=True)