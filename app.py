from flask import Flask,render_template,request
from flask_cors import cross_origin
from Preprocessing import Preprocessor
import matplotlib.pyplot as plt


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
        prediction_result = preprocessorobj.predictingTestData(test)
        result = prediction_result
        return render_template('results.html', result=result)
    except Exception as e:
        raise Exception(f"(app.py) - Something went wrong"+str(e))


@app.route("/single",methods=['POST'])
def singleForecast():
    date = request.form['content']
    try:
        preprocessorobj = Preprocessor()
        train = preprocessorobj.readDataset(path='Training_file/train_data.csv')
        fig1 = train['Petrol (USD)'].plot(figsize=(25, 8))
        plt.savefig('Plots/fig1.png')
        null_present = preprocessorobj.checkNullvalues(train)
        if null_present:
            train = preprocessorobj.fillMissingValues(train)
        outliers_present, outliers = preprocessorobj.checkOutliers(train)
        if outliers_present:
            train = preprocessorobj.replacingOutliers(train, outliers)
        fig2 = train['Petrol (USD)'].plot(figsize=(25, 8))
        plt.savefig('Plots/fig2.png')
        train = preprocessorobj.creatingNewFeatures(train)
        train = preprocessorobj.removeCorrFeatures(train)
        preprocessorobj.trainModel(train)
        test = preprocessorobj.PreparingForPrediction(date)
        prediction_result = preprocessorobj.predictingSingleDate(test)
        result = prediction_result
        return render_template('results.html', result=result)
    except Exception as e:
        raise Exception(f"(app.py) - Something went wrong"+str(e))

if __name__ == "__main__":
    app.run(port=9000,debug=True)