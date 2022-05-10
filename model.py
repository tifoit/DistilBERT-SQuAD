import json
from os import environ
from pathlib import Path

from transformers import (AutoTokenizer, AutoModelForQuestionAnswering, pipeline)

from download import download_model


def download_models(models, path):
    """
    Download models from Hugging Face repositories into a specific folder.
    If a folder already exists for a model then it is assumed it was downloaded before.
    @param path: Path where models will be stored.
    @param models: JSON objects with models to download.
    @return:A tuple of pipelines and the default one.
    """
    for model in models:
        download_model(path, model)


def load_models(path, device):
    """
    Iterates over a specific path and loads all models available.
    For each model a pipeline is created and stored in a pipelines map,
    where the key is the model name (as defined in the config.json file)
    or the folder path if that value is not present.
    @param path: Location to scan.
    @return: Tuple consisting of: a map of pipelines and a default pipeline.
    """
    print("Loading models...")
    pipelines = {}
    default_pipeline = None
    config_file = 'config.json'
    for config_file_path in Path(path).rglob(config_file):
        model_path = str(config_file_path.parent)

        name_path_key = '_name_or_path'
        with open(config_file_path) as jsonFile:
            json_object = json.load(jsonFile)
            # Looks for the file name in the config file. If not present then it will use the path as the name.
            model_name = json_object[name_path_key] if name_path_key in json_object else str(
                config_file_path.parent).replace(path, '')

        question_answer_tokenizer = AutoTokenizer.from_pretrained(model_path)
        question_answer_model = AutoModelForQuestionAnswering.from_pretrained(model_path)

        current_pipeline = pipeline('question-answering',
                                    model=question_answer_model,
                                    tokenizer=question_answer_tokenizer,
                                    device=device)

        pipelines[model_name] = current_pipeline
        # Sets the first model found as the default one.
        if default_pipeline is None:
            default_pipeline = current_pipeline
        print(f"Model loaded: $model_name")
    return pipelines, default_pipeline


class Model:

    def __init__(self, path: str):
        PATH_ENV_VARIABLE = "MODELS_PATH"
        CPU_GPU_DEVICE_VARIABLE = "CPU_GPU_DEVICE"
        # If an environment variable with MODEL_PATH has been set, then use it.
        path = environ[PATH_ENV_VARIABLE] if environ.get(PATH_ENV_VARIABLE) is not None else path
        '''
        Device ordinal for CPU/GPU supports. 
        Setting this to -1 will leverage CPU, >=0 will run the model on the associated CUDA device id.
        See https://huggingface.co/transformers/v3.0.2/main_classes/pipelines.html
        '''
        device = environ[CPU_GPU_DEVICE_VARIABLE] if environ.get(CPU_GPU_DEVICE_VARIABLE) is not None else -1
        self.pipelines, self.default_pipeline = load_models(path, device)

    def get_pipeline(self, model_name):
        """
        Chooses which pipeline to use.
        If no model_name is specific then use the default pipeline.
        if a model_name is provided but it doesn't exist, then return the default pipeline.
        @param model_name: Name of the model.
        @return: Pipeline associated to the model name.
        """
        if model_name is None or model_name not in self.pipelines.keys():
            return self.default_pipeline
        else:
            return self.pipelines[model_name]

    def predict(self, contexts, question, model_name):
        """
        Locates the answer to a question across several context texts.
        @param contexts: Possible answers.
        @param question: Posed question.
        @param model_name: Model name to use.
        @return: Object which contains the answer, its position and a score.
        """
        questions = [question] * len(contexts)  # There must be a context text per asked question
        selected_pipeline = self.get_pipeline(model_name)
        answers = selected_pipeline(question=questions, context=contexts)

        # if a single element is returned, then convert it to list
        if not isinstance(answers, list):
            answers = [answers]

        return answers
