from flask import request, Response
from model import model

import flask
import json

app = flask.Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    """Predicts an answer given some chunks of text"""
    body = request.get_json()
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


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
