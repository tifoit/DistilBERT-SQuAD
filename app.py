import os

from flask import request, Response
from model import Model

from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, Trainer

import spacy
import numpy as np
import torch

import flask
import json
import datetime

app = flask.Flask(__name__)
is_training = False
nlp = None
default_batch = 2

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


@app.route('/status', methods=['GET'])
def status():
    """
    Returns the value of is_training.
    @return: JSON response with a boolean indicating if a training is in progress.
    """
    return Response(json.dumps({
            'training_status': is_training,
    }), 200, mimetype='application/json')


@app.route('/train', methods=['POST'])
def train_vocab():
    """
    Adds new vocabulary and trains a model on the given data.
    @return: Response in JSON format.
    """

    body = request.get_json()

    if not body:
        return error_message('Missing input body', 400)
    if 'model' not in body:
        return error_message('Missing parameter model', 400)
    if 'output_path' not in body:
        return error_message('Missing parameter output_path', 400)
    if 'data' not in body:
        return error_message('Missing parameter data', 400)
    if body['data'] is None:
        return error_message('Missing information in parameter data', 400)

    global is_training
    is_training = True
    response = []

    try:

        data = body['data']
        output_path = str(body['output_path'])
        model_name = str(body['model'])

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        batch_size = body['batch_training'] if 'batch_training' in body else default_batch

        loaded_model = AutoModelForMaskedLM.from_pretrained(model_name)
        loaded_model = loaded_model.to(device)

        global nlp
        nlp = spacy.load("en_core_web_sm", exclude=['morphologizer', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

        different_tokens_list = get_new_tokens(data, tokenizer)
        new_tokens = [k for k, v in different_tokens_list]

        tokenizer.add_tokens(new_tokens)
        loaded_model.resize_token_embeddings(len(tokenizer))

        tokenizer.save_pretrained(output_path)

        train_mlm(data, loaded_model, tokenizer, output_path, batch_size)

        response = Response(json.dumps({
            'message': "MLM Training finished, model saved at: '" + output_path + "'",
            'added_tokens': new_tokens,
            'timestamp': datetime.datetime.now().isoformat()
        }), 200, mimetype='application/json')

    except Exception as e:
        response = error_message(str(e), 500)

    finally:
        is_training = False
        return response


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


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def train_mlm(docs, loaded_model, tokenizer, output_path, batch_size):
    inputs = tokenizer(docs, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    inputs['labels'] = inputs.input_ids.detach().clone()

    # create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(inputs.input_ids.shape)
    # create mask array
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
               (inputs.input_ids != 102) * (inputs.input_ids != 0)

    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

    dataset = TrainDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=15, shuffle=True)

    loaded_model.train()

    args = TrainingArguments(
        save_strategy="no",
        output_dir=output_path,
        per_device_train_batch_size=batch_size,
        num_train_epochs=2
    )

    trainer = Trainer(
        model=loaded_model,
        args=args,
        train_dataset=dataset
    )

    global is_training
    is_training = True

    trainer.train()

    loaded_model.save_pretrained(output_path)

    return


def get_new_tokens(documents, tokenizer):
    tfidf_vectorizer = TfidfVectorizer(lowercase=False, tokenizer=spacy_tokenizer, norm='l2', use_idf=True,
                                       smooth_idf=True, sublinear_tf=False)

    length = len(documents)
    tfidf_vectorizer.fit_transform(documents)

    idf = tfidf_vectorizer.idf_

    idf_sorted_indexes = sorted(range(len(idf)), key=lambda k: idf[k])
    idf_sorted = idf[idf_sorted_indexes]
    tokens_by_df = np.array(tfidf_vectorizer.get_feature_names())[idf_sorted_indexes]
    dfreqs_sorted = dfreq(idf_sorted, length).astype(np.int32)
    tokens_dfreqs = {tok: dfreq for tok, dfreq in zip(tokens_by_df, dfreqs_sorted)}
    tokens_pct_list = [int(round(dfreq / length * 100, 2)) for token, dfreq in tokens_dfreqs.items()]

    pct = 1
    index_max = len(np.array(tokens_pct_list)[np.array(tokens_pct_list) >= pct])
    new_tokens = tokens_by_df[:index_max]

    old_vocab = [k for k, v in tokenizer.get_vocab().items()]
    new_vocab = [token for token in new_tokens]

    different_tokens_list = list()

    for idx_new, w in enumerate(new_vocab):
        try:
            idx_old = old_vocab.index(w)
        except:
            idx_old = -1
        if not idx_old >= 0:
            different_tokens_list.append((w, idx_new))

    return different_tokens_list


def spacy_tokenizer(document):
    # tokenize the document with spaCY
    global nlp
    doc = nlp(document)
    # Remove stop words and punctuation symbols
    tokens = [
        token.text for token in doc if (
                token.is_stop == False and
                token.is_punct == False and
                token.text.strip() != '' and
                token.text.find("\n") == -1)]
    return tokens


def dfreq(idf, n):
    return (1 + n) / np.exp(idf - 1) - 1


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
