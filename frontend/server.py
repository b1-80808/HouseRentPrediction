from flask import Flask, request, render_template
import pickle

# load the model
with open('xgboost.pkl', 'rb') as file:
    model = pickle.load(file)


# create a flask application
app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
    # read the file contents and send them to client
    return render_template('index.html')


@app.route("/classify", methods=["POST"])
def classify():
    # get the values entered by user
    print(request.form)
    male = float(request.form.get("male"))
    age = float(request.form.get("age"))
    BPMeds = float(request.form.get("BPMeds"))
    prevalentHyp = float(request.form.get("prevalentHyp"))
    diabetes = float(request.form.get("diabetes"))
    totChol = float(request.form.get("totChol"))
    sysBP = float(request.form.get("sysBP"))
    diaBP = float(request.form.get("diaBP"))
    glucose = float(request.form.get("glucose"))

    answers = model.predict([
        [male, age, BPMeds, prevalentHyp, diabetes, totChol, sysBP, diaBP, glucose]
    ])

    if answers[0] == 1:
        return f"God may bless you!!! You will be suffering with CHD"
    else:
        return "Congrats!!! you wont be suffering with CHD"



# start the application
app.run(host="0.0.0.0", port=8000, debug=True)
