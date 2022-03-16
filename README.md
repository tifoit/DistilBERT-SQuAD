# Answer Location Service

This projects provides an Answer Location Service, whereby given a question and a text the answer is going to be located within that text.

For example:

Question: ```How many games are required to win the FA Cup?```

Text: ```The FA Cup is open to any eligible club down to Level 10 of the English football league system – 20 professional clubs in the Premier League (level 1),72 professional clubs in the English Football League (levels 2 to 4), and several hundred non-League teams in steps 1 to 6 of the National League System (levels 5 to 10). A record 763 clubs competed in 2011–12. The tournament consists of 12 randomly drawn rounds followed by the semi-finals and the final. Entrants are not seeded, although a system of byes based on league level ensures higher ranked teams enter in later rounds.  The minimum number of games needed to win, depending on which round a team enters the competition, ranges from six to fourteen```

Answer: ```six to fourteen```

The code found in this repository acts as an API built on top of a Hugging Face based model which is accessed using the [transformers library](https://github.com/huggingface/transformers). [Hugging Face](https://huggingface.co/models) models are supported, as well as custom ones (that is, not available from the Hugging Face's repository).

# Local Installation

## Download Models

### Hugging Face Models

First and foremost, you need to download models locally. For this, `init_models.py` can be executed. In here you can configure the models to download and a path where they will be stored:

```python
models = [{"model": 'deepset/roberta-base-squad2'},
          {"model": 'oliverproud/distilbert-finetuned-model'},
          {"model": 'mrm8488/distilbert-multi-finetuned-for-xqua-on-tydiqa'},
          {"model": 'distilbert-base-cased-distilled-squad'}]
path = "./models"
```
THe model name needs to match the name in the [Hugging Face repository](https://huggingface.co/models).

### Custom Models

Custom models can be installed too, however this needs to be done manually. You will need to copy your model to the expected path (as defined in variable `path`).

## Set up Environment

NOTE: Instead of following the steps below, a Python IDE such as PyCharm can be used. This will take care of most of these details.

### Python venv

In Python3 you can set up a virtual environment with

```bash
python3 -m venv /path/to/new/virtual/environment
```

Or by installing virtualenv with pip by doing
```bash
pip3 install virtualenv
```
Then creating the environment with
```bash
virtualenv venv
```
and finally activating it with
```bash
source venv/bin/activate
```

You must have Python3

Install the requirements with:
```bash
pip3 install -r requirements.txt
```

## Start Up the Service

Execute the `app.py` script. At startup, this will load all models found in a specific path. 

NOTE: This path defaults to `./models` but it can be overriden with an environment variable.


# Docker

Build the container:

```bash
docker build -t distilbert-sq uad-flask .
```

Run the container:

```bash
docker run -dp 8080:8080 -e WORKERS=2 -e THREADS=8 -e TIMEOUT=900 distilbert-squad-flask
```


