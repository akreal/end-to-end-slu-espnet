#!/usr/bin/env python3

# Copyright 2020 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import kaldiio
import numpy as np
import sys
import torch

utts = []
targets = []

with open(sys.argv[1]) as js:
    data = json.load(js)

    utts = sorted(data['utts'].keys())

    for utt in utts:
        targets.append(kaldiio.load_mat(data['utts'][utt]['output'][0]['feat']).squeeze())

predictions = []

with open(sys.argv[2]) as js:
    data = json.load(js)

    for utt in utts:
        predictions.append(np.array(data['utts'][utt]))

targets = torch.from_numpy(np.stack(targets))
predictions = torch.from_numpy(np.stack(predictions)).to(torch.float32)

print('shape {}'.format(targets.shape))
print('cosine embedding loss {}'.format(torch.nn.functional.cosine_embedding_loss(targets, predictions, torch.ones(targets.size(0)))))
print('mse loss {}'.format(torch.nn.functional.mse_loss(targets, predictions)))
print('l1 loss {}'.format(torch.nn.functional.l1_loss(targets, predictions)))
 
