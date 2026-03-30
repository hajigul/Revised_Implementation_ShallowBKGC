import os
import re
import json
import warnings
import numpy as np
import torch
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from model import ShallowBKGC

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
        filepath = "%s%s.txt" % (data_dir, data_type)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            triples = f.read().strip().split("\n")
            data = []
            for i in triples:
                parts = i.split()
                if len(parts) == 3:
                    s, p, o = parts
                    if s in entities and o in entities:
                        data.append([s, p, o])
            return data

    @staticmethod
    def load_data(data_dir, data_type="train"):
        filepath = "%s%s.txt" % (data_dir, data_type)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data if len(i.split()) == 3]
        return data

    @staticmethod
    def get_entities(data):
        entities = set()
        for i in data:
            if len(i) == 3:
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


class TestDataset:
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
        self.entity_idx = None

    def processed_data(self, dataset: Data):
        y = []
        x = []
        entitiy_idx = dict()

        print("Processing training data into subject-object pairs...")

        sub_obj_pairs = dataset.get_entity_pairs_with_predicates(dataset.train_data)
        
        for s_o_pair, predicates in sub_obj_pairs.items():
            s, o = s_o_pair
            entitiy_idx.setdefault(s, len(entitiy_idx))
            entitiy_idx.setdefault(o, len(entitiy_idx))
            x.append([entitiy_idx[s], entitiy_idx[o]])
            y.append(list(predicates))

        # Dynamic JSON filename - will be FB15KentityIDx_json
        data_path = dataset.info['dataset'].strip('/')
        kg_name = data_path.split('/')[-1] if '/' in data_path else data_path
        json_filename = f"{kg_name}entityIDx_json"
        
        entityIDx_json = json.dumps(entitiy_idx, sort_keys=False, indent=4, separators=(',', ':'))
        with open(json_filename, 'w', encoding='utf-8') as f:
            f.write(entityIDx_json)

        print(f"✅ Saved '{json_filename}' with {len(entitiy_idx)} entities")

        x = np.array(x)
        binarizer = MultiLabelBinarizer()
        y = binarizer.fit_transform(y)

        return x, y, entitiy_idx, binarizer

    def train_and_eval(self):
        self.logger.info("Info pertaining to dataset: {0}".format(self.dataset.info))
        self.logger.info("Number of triples in training data: {0}".format(len(self.dataset.train_data)))
        self.logger.info("Number of triples in validation data: {0}".format(len(self.dataset.valid_data)))
        self.logger.info("Number of triples in testing data: {0}".format(len(self.dataset.test_data)))

        self.logger.info('Data is being reformatted for multi-label classification.')
        
        X, y, self.entity_idx, binarizer = self.processed_data(self.dataset)

        self.logger.info('Building ShallowBKGC model...')
        model = ShallowBKGC(settings=self.settings, 
                           num_entities=len(self.entity_idx), 
                           num_relations=y.shape[1])

        self.logger.info('ShallowBKGC starts training.')
        model.fit(X, y)

        self.logger.info('Evaluating on test data...')
        self.eval_relation_prediction(model=model, binarizer=binarizer, triples=self.dataset.test_data)

    def eval_relation_prediction(self, model: ShallowBKGC, binarizer, triples):
        self.logger.info('Relation Prediction Evaluation begins.')

        x_ = []
        y_ = []

        for i in triples:  
            s, p, o = i
            x_.append((self.entity_idx[s], self.entity_idx[o]))
            y_.append(p)

        tensor_pred = torch.from_numpy(model.predict(np.array(x_)))
        _, ranked_predictions = tensor_pred.topk(k=len(binarizer.classes_))
        ranked_predictions = ranked_predictions.numpy()

        classes_ = binarizer.classes_.tolist()
        hits = [[] for _ in range(10)]
        ranks = []
        rank_per_relation = defaultdict(list)

        for i in range(len(y_)):
            true_relation = y_[i]
            try:
                ith_class = classes_.index(true_relation)
            except ValueError:
                continue

            rank = np.where(ranked_predictions[i] == ith_class)[0][0] + 1

            rank_per_relation[true_relation].append(rank)
            ranks.append(rank)

            for hits_level in range(10):
                if rank <= (hits_level + 1):
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

        hits = np.array(hits)
        ranks = np.array(ranks)

        self.logger.info('########## Relation Prediction Results ##########')
        self.logger.info('Mean Hits @1: {0:.4f}'.format(sum(hits[0]) / len(y_) if len(y_) > 0 else 0))
        self.logger.info('Mean Hits @3: {0:.4f}'.format(sum(hits[2]) / len(y_) if len(y_) > 0 else 0))
        self.logger.info('Mean Hits @5: {0:.4f}'.format(sum(hits[4]) / len(y_) if len(y_) > 0 else 0))
        self.logger.info('Mean rank: {0:.2f}'.format(np.mean(ranks)))
        self.logger.info('Mean reciprocal rank (MRR): {0:.4f}'.format(np.mean(1. / ranks)))
        self.logger.info('###############################################')