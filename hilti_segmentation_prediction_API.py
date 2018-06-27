import pandas as pd
import json
import dill as pickle
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():
    try:
        test_json = request.get_json(silent=True)
        json_string = json.dumps(test_json)
        testData = pd.read_json(json_string, orient='records')
        
        #preprocess data
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        le.fit(testData['ProductCategory'])
        transformedProductCategory = le.transform(testData['ProductCategory'])
        testData['transformedProductCategory'] = transformedProductCategory

        feature_names = ['AccountScore', 'Complexity', 'CustomerID', 'DeviceRating', 'EquipmentPrice', 'MonthlyMaintenance', 'ProductWeight', 'UsagePerDay']
        Xtest = testData[feature_names]
        ytest = testData['UsageType']
        print(Xtest)
        print(ytest)
    except Exception as e:
        raise e

    clf = 'model_v1.pk'

    if Xtest.empty:
        return(bad_request())
    else:
        print("Loading the model...")
        loaded_model = None
        with open(clf,'rb') as f:
            loaded_model = pickle.load(f)

        print("The model has been loaded...doing predictions now...")
        predictions = loaded_model.predict(Xtest)
        print(predictions)
        # check confusion matrix and classification report (here for )
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        
        print(confusion_matrix(ytest, predictions))
        print(classification_report(ytest, predictions))

        prediction_series = list(pd.Series(predictions))
        testData['predictions'] = prediction_series

        responses = jsonify(predictions=testData.to_json(orient="records"))
        responses.status_code = 200

        return (responses)
        
if __name__ == '__main__':
    app.run(host='localhost', debug=False, use_reloader=True)