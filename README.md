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

## Models

Models need to be downloaded to the `models` folder within the project (at the same level as the `Dockerfile`). These will be copied to the image.

NOTE: This is acceptable for local Docker testing only. When running this in a Kubernetes cluster, models will be loaded to a NFS and the pod will read them from there.


## Build the container

```bash
docker build -t distilbert-squad-flask .
```

## Run the container

```bash
docker run -dp 8080:8080 -e WORKERS=2 -e THREADS=8 -e TIMEOUT=900 distilbert-squad-flask
```

# Example Queries

## Predict
Verb: `POST`

Endpoint: `http://localhost:8080/predict`

Payload: 
```json
{
    "model": "distilbert-base-cased-distilled-squad",
    "question": "How many games are required to win the FA Cup?",
    "chunks": [
        {
            "text": "The FA Cup is open to any eligible club down to Level 10 of the English football league system – 20 professional clubs in the Premier League (level 1),72 professional clubs in the English Football League (levels 2 to 4), and several hundred non-League teams in steps 1 to 6 of the National League System (levels 5 to 10). A record 763 clubs competed in 2011–12. The tournament consists of 12 randomly drawn rounds followed by the semi-finals and the final. Entrants are not seeded, although a system of byes based on league level ensures higher ranked teams enter in later rounds.  The minimum number of games needed to win, depending on which round a team enters the competition, ranges from six to fourteen.",
            "id": "1"
        },
         {
            "text": "The first six rounds are the Qualifying Competition, from which 32 teams progress to the first round of the Competition Proper, meeting the first of the 48 professional teams from Leagues One and Two. The last entrants are the Premier League and Championship clubs, into the draw for the Third Round Proper.[2] In the modern era, only one non-League team has ever reached the quarter-finals, and teams below Level 2 have never reached the final.[note 1] As a result, significant focus is given to the smaller teams who progress furthest, especially if they achieve an unlikely \"giant-killing\" victory.",
            "id": "2"
        }
    ]
}
```

Sample Response:
```json
[
    {
        "score": 0.5027955770492554,
        "start": 693,
        "end": 708,
        "answer": "six to fourteen",
        "id": "1",
        "highlight": "The FA Cup is open to any eligible club down to Level 10 of the English football league system – 20 professional clubs in the Premier League (level 1),72 professional clubs in the English Football League (levels 2 to 4), and several hundred non-League teams in steps 1 to 6 of the National League System (levels 5 to 10). A record 763 clubs competed in 2011–12. The tournament consists of 12 randomly drawn rounds followed by the semi-finals and the final. Entrants are not seeded, although a system of byes based on league level ensures higher ranked teams enter in later rounds.  The minimum number of games needed to win, depending on which round a team enters the competition, ranges from <span class=\"highlight\">six to fourteen</span>."
    },
    {
        "score": 0.15806056559085846,
        "start": 10,
        "end": 13,
        "answer": "six",
        "id": "2",
        "highlight": "The first <span class=\"highlight\">six</span> rounds are the Qualifying Competition, from which 32 teams progress to the first round of the Competition Proper, meeting the first of the 48 professional teams from Leagues One and Two. The last entrants are the Premier League and Championship clubs, into the draw for the Third Round Proper.[2] In the modern era, only one non-League team has ever reached the quarter-finals, and teams below Level 2 have never reached the final.[note 1] As a result, significant focus is given to the smaller teams who progress furthest, especially if they achieve an unlikely \"giant-killing\" victory."
    }
]
```

## Models
Verb: `GET`

Endpoint: `http://localhost:8080/models`

Payload: None

Sample Result:
```json
[
    "deepset/roberta-base-squad2",
    "distilbert-base-cased-distilled-squad",
    "mrm8488/distilbert-multi-finetuned-for-xqua-on-tydiqa",
    "models\\oliverproud\\distilbert-finetuned-model"
]
```
