import os

from flask import request, Response
from model import Model

import flask
import json
import datetime

app = flask.Flask(__name__)

'''
A path is passed when creating models. This can also be overriden as a environmental variable.
'''
model = Model(path="./models")


@app.route('/models', methods=['GET'])
def models():
    """
    Lists all available models.
    @return: JSON response with a list of models.
    """
    response = []
    for pipeline_name in model.pipelines:
        response.append(pipeline_name)

    return Response(json.dumps(response), mimetype='application/json')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts an answer given some chunks of text.
    @return: Response in JSON format.
    """
    body = request.get_json()

    if not body:
        return error_message('Missing input body', 400)

    if 'question' not in body:
        return error_message('The prediction needs a question', 400)

    if 'chunks' not in body:
        return error_message('The prediction needs at least one chunk', 400)

    question = str(body['question'])
    chunks = body['chunks']
    # Checks if a specific model name has been supplied, if it sets it to None to use the default one
    model_name = body['model'] if 'model' in body else None
    style = body['style'] if 'style' in body else 'highlight'
    response = []

    texts = [chunk['text'] for chunk in chunks]
    # Gets predictions for all texts at once.
    predictions = model.predict(texts, question, model_name)
    for index, prediction in enumerate(predictions):
        chunk = chunks[index]
        prediction['id'] = chunk['id']

        if 'answer' in prediction and prediction['answer']:
            prediction['highlight'] = highlight(chunk['text'], prediction, style)

        response.append(prediction)

    return Response(json.dumps(sorted(response, key=lambda answer: answer['score'], reverse=True)),
                    mimetype='application/json')


def highlight(text, answer, style):
    """
    Highlights the answer in the given text
    @param text: Context where answer is located.
    @param answer: Located answer.
    @param style: Highlighting option.
    @return: Returns text with highlighted answer.
    """

    start_pos = answer['start']
    end_pos = answer['end']

    pre = text[:start_pos]
    text_to_highlight = text[start_pos:end_pos]
    post = text[end_pos:]

    return '{}<span class="{}">{}</span>{}'.format(pre, style, text_to_highlight, post)


def error_message(message, status):
    """Builds a JSON response with an error message"""
    return Response(json.dumps({
        'message': message,
        'status': status,
        'timestamp': datetime.datetime.now().isoformat()
    }), status, mimetype='application/json')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
