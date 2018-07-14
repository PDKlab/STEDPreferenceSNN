import json

import numpy as np
import requests

class VirtualNet(object):

    def __init__(self, address, port=5000):
        self.address = address
        self.port = port
        self.url = 'http://{}:{}'.format(self.address, self.port)

    def predict(self, pair_set):
        pair_set2send = json.dumps({'pair_set':pair_set.astype(float).tolist(), 
                               'type':'{}'.format(pair_set.dtype)})
        r = requests.post(self.url, data=pair_set2send)
        return json.loads(r.text)['good_pair']