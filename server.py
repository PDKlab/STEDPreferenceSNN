import os
import json
import argparse

import torch
import numpy as np
from flask import Flask
from flask import render_template, request

from src.models import PrefNet


global model, mean, std

## creating app
app = Flask(__name__)

@app.route("/", methods=['POST'])
def get_good_pair():
    data = json.loads(request.data.decode('utf-8'))
    pair_set = np.array(data['pair_set']).astype(data['type'])
    pair_set -= mean
    pair_set /= std
    good_pair = int(np.argmax(model.predict(pair_set)))
    return json.dumps({'good_pair':good_pair}), 200, {'ContentType':'application/json'}

if __name__ == '__main__':
    global model, mean, std

    parser = argparse.ArgumentParser(description='Server for qualityNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', help='port if using virtual net', type=int, default=5000)
    parser.add_argument('experiment', help='experiment path', type=str)
    args = parser.parse_args()

    xp_folder = args.experiment
    with open(os.path.join(xp_folder, 'config.json'), 'r') as f:
        config = json.load(f)

    mean = config['train_mean']
    std = config['train_std']

    model = PrefNet(nb_obj=3)
    model.load_state_dict(torch.load(os.path.join(xp_folder, 'weights.t7')))
    model.eval()

    app.run(host='0.0.0.0', port=args.port, debug=False, use_reloader=False)




