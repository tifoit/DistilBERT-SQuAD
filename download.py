from pathlib import Path
import requests
import os

from transformers import AutoTokenizer, AutoModelForQuestionAnswering


def download_model(path, model):
    """
    Downloads models from Hugging Face is the 'download' flag has been set to true.
    If a folder with the model name already exists, then takes no action.
    Otherwise, the model should exist on disk.
    @param path: Path to where the models will be downloaded.
    @param model: Model object.
    @return: Path to the model on disk.
    """
    model_name = model['model']
    model_path = "{}/{}/".format(path, model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)  # Create the dir
        print("Downloading model {} to {}".format(model_name, model_path))
        download_from_huggingface(model_name, model_path)
    else:
        print("Model already exists: {}".format(model_path))
    return model_path


def download_from_huggingface(model_name, model_path):
    """
    To download files related to the model and the tokenizer, first it is needed to instantiate the model and the tokenizer,
    which will download the files from Hugging Face's repo. Then these are saved to disk.
    @param model_name: Name of the model (as expected to be found in Hugging Face)
    @param model_path: Path to where to store the model on disk
    @return:
    """
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.save_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_path)
