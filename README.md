# DistilBERT-SQuAD

# WIP  🚧

# The Stanford Question Answering Dataset (SQuAD)

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

https://rajpurkar.github.io/SQuAD-explorer/

# What is DistilBERT?

Thanks to the brilliant people at [Hugging Face](https://huggingface.co/) we now have DistilBERT, which stands for Distilated-BERT. DistilBERT is a small, fast, cheap and light Transformer model based on Bert architecture. It has 40% less parameters than `bert-base-uncased`, runs 60% faster while preserving 97% of BERT's performance as measured on the GLUE language understanding benchmark. DistilBERT is trained using knowledge distillation, a technique to compress a large model called the teacher into a smaller model called the student. By distillating Bert, we obtain a smaller Transformer model that bears a lot of similarities with the original BERT model while being lighter, smaller and faster to run. DistilBERT is thus an interesting option to put large-scaled trained Transformer model into production. [Transformers - Hugging Face repository](https://github.com/huggingface/transformers)

Victor Sahn of Hugging Face [wrote a great Medium post](https://medium.com/huggingface/distilbert-8cf3380435b5) introducing DistilBERT and explaining parts of their newly released [NeurIPS 2019 Workshop paper](https://arxiv.org/abs/1910.01108)

# Installation

If you are testing this on your own machine I would recommend you do the setup in a virtual environment, as not to affect the rest of your files. 

In Python3 you can set up a virtual environment with `python3 -m venv /path/to/new/virtual/environment`. Or by installing virtualenv with pip by doing `pip3 install virtualenv`, creating the environment with `virtualenv venv`, and finally activating it with `source venv/bin/activate`

You must have Python3

Install the requirements with:

`pip3 install -r requirements.txt`

### SQuAD Fine-tuned model 

The SQuAD fine-tuned model is available in my [S3 Bucket](https://distilbert-finetuned-model.s3.eu-west-2.amazonaws.com/pytorch_model.bin) or alternatively inside the model.py file you can specify the type of model you wish to use, the one I have provided, or a Hugging Face fine-tuned SQuAD model

`distilbert-base-uncased-distilled-squad`. 

You can do this with 

`model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad', config=config)`

# Making predictions

<iframe
  src="https://carbon.now.sh/embed/?bg=rgba(0%2C138%2C255%2C1)&t=lucario&wt=none&l=python&ds=true&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=false&fl=1&fm=Hack&fs=14px&lh=133%25&si=false&es=2x&wm=false&code=from%2520model%2520import%2520Model%250A%250Amodel%2520%253D%2520Model('model')%250A%250Acontext%2520%253D%2520%2522Netflix%2520uses%2520a%2520variety%2520of%2520methods%2520to%2520help%2520you%2520find%2520TV%2520shows%2520and%2520movies%2520to%2520enjoy.%2520%255C%250A%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520You%2520can%2520find%2520TV%2520shows%2520and%2520movies%2520through%2520Recommendations%2520or%2520Search%252C%2520or%2520by%2520browsing%2520through%2520categories.%2522%250A%250Aquestion%2520%253D%2520%2522How%2520do%2520I%2520find%2520TV%2520shows%2520and%2520movies%2520on%2520Netflix%253F%2522%250A%250Aanswer%2520%253D%2520model.predict(context%252C%2520question)%250A%250Aprint(%2522Question%253A%2520%2522%2520%252B%2520question)%250Aprint(%2522Answer%253A%2520%2522%2520%252B%2520answer%255B%2522answer%2522%255D)%250A"
  style="transform:scale(0.7); width:1024px; height:473px; border:0; overflow:hidden;"
  sandbox="allow-scripts allow-same-origin">
</iframe>
