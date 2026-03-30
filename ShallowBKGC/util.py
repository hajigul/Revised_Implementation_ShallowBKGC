import datetime
import os
import pickle
import numpy as np
import time
import bz2
import re
import logging
from scipy.sparse import csr_matrix
from scipy import stats
import pandas as pd

triple = 3


def create_experiment_folder(folder_name='Experiments'):
    """Safe version that works on Windows"""
    directory = os.path.join(os.getcwd(), folder_name)
    safe_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_of_folder = os.path.join(directory, safe_time)
    
    os.makedirs(path_of_folder, exist_ok=True)
    return path_of_folder, directory


def create_logger(*, name, p):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(os.path.join(p, 'info.log'))
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def performance_debugger(func_name):
    def function_name_decoratir(func):
        def debug(*args, **kwargs):
            starT = time.time()
            print('\n\n######', func_name, ' starts ######')
            r = func(*args, **kwargs)
            print(func_name, ' took ', time.time() - starT, ' seconds\n')
            return r
        return debug
    return function_name_decoratir


def pairwise_iteration(it):
    it = iter(it)
    while True:
        yield next(it), next(it)


def get_path_knowledge_graphs(path: str):
    KGs = list()
    if os.path.isfile(path):
        KGs.append(path)
    else:
        for root, dir_, files in os.walk(path):
            for file in files:
                if '.nq' in file or '.nt' in file or 'ttl' in file:
                    KGs.append(os.path.join(path, file))
    if len(KGs) == 0:
        print(path + ' is not a path for a file or a folder containing any .nq or .nt formatted files')
        exit(1)
    return KGs


def file_type(f_name):
    if str(f_name).lower().endswith('.bz2'):
        return bz2.open(f_name, "rt")
    return open(f_name, "r")


def serializer(*, object_: object, path: str, serialized_name: str):
    with open(os.path.join(path, serialized_name + ".p"), "wb") as f:
        pickle.dump(object_, f)


def deserializer(*, path: str, serialized_name: str):
    with open(os.path.join(path, serialized_name + ".p"), "rb") as f:
        return pickle.load(f)


@performance_debugger('Training')
def learn(model, storage_path, x, y, batch_size=10000, epochs=1):
    history = model.fit(x, y, batch_size=batch_size, epochs=epochs, use_multiprocessing=True)
    return model, history


# ====================== Unused / Legacy functions ======================
def recall_m(y_true, y_pred):
    from tensorflow.keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    from tensorflow.keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def decompose_rdf(sentence):
    components = sentence.strip().split()
    if len(components) != 3:
        raise ValueError(f"Unsupported RDF format: {sentence}")
    s, p, o = components
    s = re.sub(r"\s+", "", s)
    p = re.sub(r"\s+", "", p)
    o = re.sub(r"\s+", "", o)
    return s, p, o, 3