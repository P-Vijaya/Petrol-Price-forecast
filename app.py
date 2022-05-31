from flask import Flask,render_template,request
from flask_cors import cross_origin
from Preprocessing import Preprocessor
import matplotlib.pyplot as plt
import os
from wsgiref import simple_server

app = Flask(__name__)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/test",methods=['POST'])
def testForecast():
    try:
        preprocessorobj = Preprocessor()
        train = preprocessorobj.readDataset(path='Training_file/train_data.csv')
        fig1 = train['Petrol (USD)'].plot(figsize=(25, 8))
        plt.savefig('Plots/fig1.png')
        test = preprocessorobj.readDataset(path='Testing_file/test_data.csv')
        null_present = preprocessorobj.checkNullvalues(train)
        if null_present:
            train = preprocessorobj.fillMissingValues(train)
        outliers_present, outliers = preprocessorobj.checkOutliers(train)
        if outliers_present:
            train = preprocessorobj.replacingOutliers(train, outliers)
        fig2 = train['Petrol (USD)'].plot(figsize=(25, 8))
        plt.savefig('Plots/fig2.png')
        train = preprocessorobj.creatingNewFeatures(train)
        test = preprocessorobj.creatingNewFeatures(test)
        test = test.drop(['Prediction'], axis=1)
        train = preprocessorobj.removeCorrFeatures(train)
        ## removing multicollinear features in test data
        test = test.drop(['dayofyear', 'weekofyear'], axis=1)
        preprocessorobj.trainModel(train)
        prediction_result = preprocessorobj.predictingTestData(data=test,filename='test_data.csv')
        result = prediction_result
        print("***********************")
        print(type(result))
        print(result)
        print("***********FOR LOOP************")
        date = result['Date']
        price = result['Prediction']
        results = []
        for i in range(len(date)):
            mydict = {'Date': date[i], 'Price': price[i]}
            results.append(mydict)
        print("*************END LOOP**********")
        return render_template('results.html', results=results)
    except Exception as e:
        raise Exception(f"(app.py) - Something went wrong"+str(e))


@app.route("/single",methods=['POST'])
def singleForecast():
    file = request.files['Filename']
    file.save(os.path.join(r'C:\Users\VIMALA P T\OneDrive\Documents\Python anaconda\Ineuron\projects\Ineuron Internship\Petrol price forecasting\Petrol Price Forecasting\Testing_file',file.filename))
    try:
        preprocessorobj = Preprocessor()
        train = preprocessorobj.readDataset(path='Training_file/train_data.csv')
        fig1 = train['Petrol (USD)'].plot(figsize=(25, 8))
        plt.savefig('Plots/fig1.png')
        test = preprocessorobj.readDataset(path='Testing_file/{}'.format(file.filename))
        null_present = preprocessorobj.checkNullvalues(train)
        if null_present:
            train = preprocessorobj.fillMissingValues(train)
        outliers_present, outliers = preprocessorobj.checkOutliers(train)
        if outliers_present:
            train = preprocessorobj.replacingOutliers(train, outliers)
        fig2 = train['Petrol (USD)'].plot(figsize=(25, 8))
        plt.savefig('Plots/fig2.png')
        train = preprocessorobj.creatingNewFeatures(train)
        test = preprocessorobj.creatingNewFeatures(test)
        train = preprocessorobj.removeCorrFeatures(train)
        test = test.drop(['Prediction','dayofyear', 'weekofyear'], axis=1)
        preprocessorobj.trainModel(train)
        #test = preprocessorobj.PreparingForPrediction(date)
        prediction_result = preprocessorobj.predictingTestData(data=test,filename=file.filename)
        result = prediction_result
        print("***********FOR LOOP************")
        date = result['Date']
        price = result['Prediction']
        results = []
        for i in range(len(date)):
            mydict = {'Date': date[i], 'Price': price[i]}
            results.append(mydict)
        print("*************END LOOP**********")
        return render_template('results.html', results=results)
    except Exception as e:
        raise Exception(f"(app.py) - Something went wrong"+str(e))

if __name__ == '__main__':
    app.run(port=9000,debug=True)