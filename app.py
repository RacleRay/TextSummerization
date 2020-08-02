#-*- coding:utf-8 -*-
# author: Racle
# project: autosummarization

from flask import Flask, request, jsonify, render_template
from embeddingmodel import EmbeddingModel


app = Flask(__name__)
model = EmbeddingModel()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def extract():
    '''
    For rendering results on HTML GUI
    '''
    raw_content = request.form.get('content')
    title = request.form.get('title')
    if len(title) == 0 or title == ' ':
        title = None
    # print(raw_content)
    result = model.summary(raw_content, title)
    return render_template('index.html', summary=result[0], keywords=result[1])


@app.route('/predict_api',methods=['POST'])
def extract_api():
    raw_content = request.get_json(force=True)
    result = model.summary(raw_content.values())
    return jsonify(result)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=9000, debug=True)