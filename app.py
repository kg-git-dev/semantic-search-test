# app.py

from flask import Flask
from api_routes import api_blueprint

app = Flask(__name__)
app.register_blueprint(api_blueprint)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == "__main__":
    app.run(port=8000, debug=False)