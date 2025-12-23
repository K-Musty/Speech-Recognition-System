from flask import Flask


app = Flask(__name__)


@app.route("/predict", methods=['POST'])





if __name__ == ("__main__"):
	app.run(debug=True)
