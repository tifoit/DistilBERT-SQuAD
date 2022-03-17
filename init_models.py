"""
NOTE: This file is only meant to run locally to download models!
"""
import model

models = [{"model": 'deepset/roberta-base-squad2'},
          {"model": 'mrm8488/distilbert-multi-finetuned-for-xqua-on-tydiqa'},
          {"model": 'distilbert-base-cased-distilled-squad'}]
path = "./models"

model.download_models(models, path)
