from flask import request, Response
from model import model

import flask
import json
import datetime

app = flask.Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    """Predicts an answer given some chunks of text"""
    body = request.get_json()

    if not body:
        return error_message('Missing input body', 400)

    if 'question' not in body:
        return error_message('The prediction needs a question', 400)

    if 'chunks' not in body:
        return error_message('The prediction needs at least one chunk', 400)

    question = str(body['question'])
    chunks = body['chunks']
    style = body['style'] if 'style' in body else 'highlight'
    response = []

    for chunk in chunks:
        text = str(chunk['text'])
        prediction = model.predict(text, question)
        prediction['id'] = chunk['id']

        if 'answer' in prediction and prediction['answer']:
            prediction['highlight'] = highlight(text, prediction['answer'], style)

        response.append(prediction)

    return Response(json.dumps(response), mimetype='application/json')


def highlight(text, answer, style):
    """Highlights the answer in the given text"""
    left_index = text.lower().find(answer.lower())

    if left_index > -1:
        right_index = left_index + len(answer)
        pre = text[:left_index]
        inner = text[left_index:right_index]
        post = text[right_index:]
        return pre + '<span class="' + style + '">' + inner + '</span>' + post
    else:
        return text


def error_message(message, status):
    """Builds a JSON response with an error message"""
    return Response(json.dumps({
        'message': message,
        'status': status,
        'timestamp': datetime.datetime.now().isoformat()
    }), status, mimetype='application/json')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
