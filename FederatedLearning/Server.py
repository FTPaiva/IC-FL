import flwr as fl
import sys
import numpy as np
from typing import List, Tuple
from flwr.common import Metrics
import time


strategy = fl.server.strategy.FedAvg(
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5
)

# Lista para nomes de arquivos
names = ['VGG', 'ResNet', 'Xception', 'MobileNet']

rep = 0
while (rep <= 3): # Executa uma vez para cada modelo a ser testado

    start_time = time.time()

    # Inicia o servidor Flower
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy = strategy
    )
    with open('FederatedLearning/history/time.txt', 'a') as f:
            f.writelines(names[rep] + ' - ' + str(time.time() - start_time) + '\n')