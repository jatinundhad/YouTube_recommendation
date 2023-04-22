from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello_world():
    return """<img src='https://i.ytimg.com/vi/kzwfHumJyYc/default.jpg'/>"""


app.run(debug=True)
