import os
import time
import datetime
import json
import pickle as pkl
import random


def key_filter(fulldict, blacklist):
    if blacklist is None:
        blacklist = []
    filtereddict = {}
    for k in fulldict.keys():
        if k not in blacklist:
            filtereddict[k] = fulldict[k]
    return filtereddict


class Experiment(object):

    def __init__(self, config, results_path=None, blacklist=None):
        ## definitions
        self.config = key_filter(config, blacklist)
        if results_path is None:
            results_path = '.'
        ## keeping timestamp
        self.timestamp = time.time()
        self.config['timestamp'] = self.timestamp
        self.ftimestamp = datetime.datetime.fromtimestamp(self.timestamp
            ).strftime('%Y-%m-%d-%H-%M-%S')
        self.root = os.path.join(results_path, self.ftimestamp)
        ## create the experiment folder
        try:
            os.mkdir(self.root)
        except FileExistsError:
            self.root = os.path.join(results_path, self.ftimestamp + '-' +\
                ''.join([str(random.randint(10)) for i in range(6)]))
        self.figuredir = os.path.join(self.root, 'figures')
        os.mkdir(self.figuredir)
        ## define names
        self.config_path = os.path.join(self.root, 'config.json')
        self.model_path = os.path.join(self.root, 'weights.t7')
        self.results_path = os.path.join(self.root, 'results.json')
        self.record_path = os.path.join(self.root, 'record.json')
        ## saving config 
        self.save_config()

    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)

    def save_model(self, func, weights):
        func(weights, self.model_path)

    def save_results(self, results, blacklist=None):
        results2save = key_filter(results, blacklist)
        with open(self.results_path, 'wb') as f:
            pkl.dump(results2save, f)

    def save_record(self, record, blacklist=None):
        record2save = key_filter(record, blacklist)
        with open(self.record_path, 'wb') as f:
            pkl.dump(record2save, f)

    def rootdir(self):
        return self.root

    def get_figuredir(self):
        return self.figuredir