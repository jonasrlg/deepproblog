import os
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
import sys
sys.path.append('../../../')

from json import dumps

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision.transforms import transforms
from typing import List

from deepproblog.dataset import Dataset, DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix, get_nn_accuracy
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

def get_accuracy(model: Model, dataset: Dataset, dataloader: List[TorchDataLoader], program_accuracy: List[float], 
                 nn_accuracy: List[List[float]]):
    program_acc = get_confusion_matrix(model, dataset).accuracy()
    program_accuracy.append(program_acc)
    nn_acc = get_nn_accuracy(model, dataloader)
    for nn, acc in zip(nn_accuracy, nn_acc):
        nn.append(acc)
    print(f'Accuracy - Program = {program_accuracy} / Neural Networks = {nn_accuracy}')
    return program_acc, nn_acc

method = "exact"
N = 1
epochs = 5
batch_size = 1_000
learning_rate = 1e-3
verbose = 1

name = "addition_{}_{}".format(method, N)

train_set = addition(N, "train")
test_set = addition(N, "test")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])
testLoader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root=dir_path+'/data/', train=False, transform=transform), batch_size=1000)

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
accuracy_program = []
accuracy_nn = [[] for _ in model.networks]

train = train_model(model, loader, epochs, log_iter=1, test_iter=1, verbose=verbose, profile=0,
                    test=lambda x: [("Accuracy", get_accuracy(x, test_set, [testLoader], accuracy_program, accuracy_nn))]
                    )
"""
model.save_state("snapshot/" + name + ".pth")
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
)
train.logger.write_to_file("log/" + name)
"""

np.save('addition_accuracy_program.npy', np.array(accuracy_program))
np.save('addition_accuracy_nn.npy', np.array(accuracy_nn[0]))

