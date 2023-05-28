import os
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
import sys
sys.path.append('../../../')

from json import dumps

import numpy as np
import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

method = "exact"
N = 1
epochs = 5
batch_size = 1_000
learning_rate = 1e-3

name = "addition_{}_{}".format(method, N)

train_set = addition(N, "train")
test_set = addition(N, "test")

network = MNIST_Net()

pretrain = 0
if pretrain is not None and pretrain > 0:
    network.load_state_dict(torch.load("models/pretrained/all_{}.pth".format(pretrain)))
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

model = Model("models/addition.pl", [net])
if method == "exact":
    model.set_engine(ExactEngine(model), cache=True)
elif method == "geometric_mean":
    model.set_engine(
        ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False)
    )

model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

loader = DataLoader(train_set, batch_size, False)
queries = test_set.to_queries()
test_queries = [q.variable_output() for q in queries]
print(f"Dataset size = {loader.length} / Batch size = {loader.batch_size}")
print(f"Queries size = {len(queries)}")
print(f'Start training for {epochs} epoch(s)...')
train = train_model(model, loader, epochs, queries, test_queries, log_iter=1, verbose=1, profile=0)
train.save_accuracy('addition.npy')
#model.save_state("snapshot/" + name + ".pth")
#train.logger.comment(dumps(model.get_hyperparameters()))
#train.logger.comment(
#    "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
#)
#train.logger.write_to_file("log/" + name)
