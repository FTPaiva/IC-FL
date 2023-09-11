import subprocess
import time
subprocess.run(['python', 'FederatedLearning/Server.py'])

time.sleep(3)

client = 1
while (client <= 5):
    subprocess.run(['python', 'FederatedLearning/Client.py', str(client)])
    client += 1