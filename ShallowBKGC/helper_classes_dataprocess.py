import os
import re
from collections import Counter, defaultdict
import itertools

from torch import device

import util as ut
import os.path
from numpy import linalg as LA
import numpy as np
import pandas as pd
import warnings
import sys
from abc import ABC, abstractmethod
from sklearn.preprocessing import MultiLabelBinarizer
from model import ShallowBKGC
import torch
import time
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

from util import *


class Data: 

    def __init__(self, data_dir=None):

        self.info = {'dataset': data_dir}
        self.train_data = self.load_data(data_dir, "train")
        self.valid_data = self.load_data_with_checking(data_dir, data_type="valid",
                                                       entities=self.get_entities(self.train_data))
        self.test_data = self.load_data_with_checking(data_dir, data_type="test",
                                                      entities=self.get_entities(self.train_data))


    @staticmethod
    def load_data_with_checking(data_dir, entities, data_type="train"):
        assert entities
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            triples = f.read().strip().split("\n")
            data = []
            for i in triples:
                s, p, o = tuple(i.split())
                if s in entities and o in entities:
                    data.append([s, p, o])
            return data


    @staticmethod
    def load_data(data_dir, data_type="train"):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
        return data

    @staticmethod
    def get_entities(data):
        entities = set()
        for i in data:
            s, p, o = i
            entities.add(s)
            entities.add(o)
        return sorted(list(entities))

    @staticmethod
    def get_entity_pairs_with_predicates(triples):
        sub_obj_pairs = dict()
        for s, p, o in triples:
            sub_obj_pairs.setdefault((s, o), set()).add(p)
        return sub_obj_pairs


class TestDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]

class Experiment:
    def __init__(self, d, s):
        self.dataset = d
        self.settings = s
        self.model_name = self.settings['model_name']
        self.storage_path, _ = create_experiment_folder()
        self.logger = create_logger(name=self.model_name, p=self.storage_path)
        self.entity_idx = None  # will be filled.

    def processed_data(self, dataset: Data):
        """

        :type dataset: object
        """
        y = []
        x = []
        entitiy_idx = dict()


        sub_obj_pairs = dataset.get_entity_pairs_with_predicates(dataset.train_data)
        for s_o_pair, predicates in sub_obj_pairs.items():
            s, o = s_o_pair
            entitiy_idx.setdefault(s, len(entitiy_idx))
            entitiy_idx.setdefault(o, len(entitiy_idx))
            x.append([entitiy_idx[s], entitiy_idx[o]])
            y.append(list(predicates))

            # save entity_idx to local
            #import json
            entityIDx_json = json.dumps(entitiy_idx, sort_keys=False, indent=4, separators=(',', ':'))
            f = open('FB15K237entityIDx_json', 'w')
            #f = open('WN18RRentityIDx_json', 'w')
            #f = open('YAGO3-10entityIDx_json', 'w')
            f.write(entityIDx_json)

        x = np.array(x)

        binarizer = MultiLabelBinarizer()
        y = binarizer.fit_transform(y)

        return x, y, entitiy_idx, binarizer






    def train_and_eval(self):
        self.logger.info("Info pertaining to dataset:{0}".format(self.dataset.info))
        self.logger.info("Number of triples in training data:{0}".format(len(self.dataset.train_data)))
        self.logger.info("Number of triples in validation data:{0}".format(len(self.dataset.valid_data)))
        self.logger.info("Number of triples in testing data:{0}".format(len(self.dataset.test_data)))

        self.logger.info('Data is being processing.')
        X, y, self.entity_idx, binarizer = self.processed_data(self.dataset)




