from flask import Flask, render_template, request
import mlflow
from preprocessing_utility import normalize_text
import pickle
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/write2shivamgithub/mlops-mini-project.mlflow")
dagshub.init(repo_owner='write2shivamgithub', repo_name='mlops-mini-project', mlflow=True)


app = Flask(__name__)

# load model from model registry
model_name = "my_model"
model_version = 3

model_uri = f'models:/{model_name}/{model_version}'

model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    text = request.form['text']

    # clean
    text = normalize_text(text)
    
    # BOW
    features = vectorizer.transform([text])

    # prediction
    result = model.predict(features)

    # show
    return  str(result[0])

    # model
    return render_template('index.html', result=result[0])


    return text
app.run(debug=True)