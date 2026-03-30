import os
import torch
import numpy as np
from enum import Enum
from typing import Tuple, List, Dict
from torch.utils.data import Dataset


class BatchType(Enum):
    HEAD_BATCH = 0
    TAIL_BATCH = 1
    SINGLE = 2


class ModeType(Enum):
    TRAIN = 0
    VALID = 1
    TEST = 2


class DataReader(object):
    def __init__(self, data_path: str):
        entity_dict_path = os.path.join(data_path, 'entities.dict')
        relation_dict_path = os.path.join(data_path, 'relations.dict')
        train_data_path = os.path.join(data_path, 'train.txt')
        valid_data_path = os.path.join(data_path, 'valid.txt')
        test_data_path = os.path.join(data_path, 'test.txt')

        self.entity_dict = self.read_dict(entity_dict_path)
        self.relation_dict = self.read_dict(relation_dict_path)

        self.train_data = self.read_data(train_data_path, self.entity_dict, self.relation_dict)
        self.valid_data = self.read_data(valid_data_path, self.entity_dict, self.relation_dict)
        self.test_data = self.read_data(test_data_path, self.entity_dict, self.relation_dict)

    def read_dict(self, dict_path: str):
        """
        Read entity / relation dict.
        Format: dict({id: entity / relation})
        """

        element_dict = {}
        with open(dict_path, 'r') as f:
            for line in f:
                id_, element = line.strip().split('\t')
                element_dict[element] = int(id_)

        return element_dict

    def read_data(self, data_path: str, entity_dict: Dict[str, int], relation_dict: Dict[str, int]):
        """
        Read train / valid / test data.
        """
        triples = []
        with open(data_path, 'r') as f:
            for line in f:
                head, relation, tail = line.strip().split('\t')
                triples.append((entity_dict[head], relation_dict[relation], entity_dict[tail]))
        return triples

class TestDataset(Dataset):
    def __init__(self, data_reader: DataReader, mode: ModeType, batch_type: BatchType):
        self.triple_set = set(data_reader.train_data + data_reader.valid_data + data_reader.test_data)
        if mode == ModeType.VALID:
            self.triples = data_reader.valid_data
        elif mode == ModeType.TEST:
            self.triples = data_reader.test_data

        self.len = len(self.triples)

        self.num_entity = len(data_reader.entity_dict)
        self.num_relation = len(data_reader.relation_dict)

        self.mode = mode
        self.batch_type = batch_type

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.batch_type == BatchType.HEAD_BATCH:
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.num_entity)]
            tmp[head] = (0, head)
        elif self.batch_type == BatchType.TAIL_BATCH:
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.num_entity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch type {} not supported'.format(self.mode))

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.batch_type

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
