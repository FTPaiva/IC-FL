## Base de dados utilizada

A base de dados escolhida foi a [Road Pothole Images for Pothole detection](https://www.kaggle.com/datasets/sovitrath/road-pothole-images-for-pothole-detection/) disponível no Kaggle.
Ela possui 3777 imagens, sendo 1119 imagens com buracos e 2658 sem buracos, todas captadas por uma câmera veicular, no estilo *dashcam* que é um modelo de câmera que acopla-se ao painel do automóvel, registrando a visão do motorista. Vale notar que todas as imagens foram capturadas na África do Sul e possuem características específicas da região, como coloração da terra, tipo de vegetação, condições de iluminação, tipo de calçada, tamanho e abundância de prédios, etc.

A pasta destinada às imagens de teste do dataset do kaggle não possui indicação de quais imagens contêm buracos, então recomendo dividir o conjunto de imagens de treinamento em 2: Um maior para treinamento e um menor para testes.

## Redes neurais utilizadas

> [!NOTE]
> Todas as redes utilizadas foram pré-treinadas com o dataset ImageNet.

- [**VGG-16**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16) (tensorflow.keras.applications.VGG16) (138,4M parâmetros)

- [**ResNet50**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/ResNet50V2) (tensorflow.keras.applications.ResNet50V2) (25,6M parâmetros)

- [**Xception**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception/Xception) (tensorflow.keras.applications.Xception) (22,9M parâmetros)

- [**MobileNetV3**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Large) (tensorflow.keras.applications.MobileNetV3Large) (5,4M parâmetros, voltada para dispositivos móveis)

## Sobre o repositório
O repositório está dividido em 2 pastas principais, [MainTraining](MainTraining) e [FederatedLearning](FederatedLearning), sendo a primeira uma versão sem a utilização do aprendizado federado.

### Pasta MainTraining
O script principal ([Classification.py](MainTraining/Classification.py)) usa os dados contidos na pasta [traindata](MainTraining/traindata), onde as imagens **positivas** (contendo buracos) ficam em uma pasta e as **negativas** em outra.

A divisão dos datasets de treinamento e validação é feita através
de uma função do keras (*tf.keras.preprocessing.image_dataset_from_directory*). Nos parâmetros dessa função, defini que 20% das imagens seriam destinadas à validação e o *batch size* seria de 32 (esses valores pode ser alterados a fim de otimização, mas utilizei valores que vi sendo recomendados). O tamanho de imagem de 224x224 foi utilizado pois a maioria das redes neurais pré-treinadas que usei tinham esse formato de input. Foi utilizada uma *seed* fixa para que os resultados sejam consistentes e as comparações feitas entre os modelos sejam mais justas.

> [!NOTE]
> Para algumas seeds, a acurácia de validação acabou ficando mais alta que a acurácia de treinamento, algo anormal. Isso pode ser um indicativo de uma divisão ruim do dataset, então algo que pode ser feito para evitar esse tipo de situação é realizar validação cruzada.

Há um **while loop** que engloba o resto do script que executa uma vez para cada rede a ser treinada, mudando partes do modelo a cada ciclo. Primeiramente, é definido qual rede pré-treinada deve ser importada, juntamente com sua função de pré-processamento específica (que prepara as imagens para se adequarem aos inputs esperados pela rede). É importante definir o parâmetro *include_top* da rede como **False**, pois a fim de realizar o *transfer learning*, alteramos o final da rede neural, então não desejamos importar as últimas camadas.

O trecho abaixo impede que as camadas do modelo base de sejam alteradas, essencial para a técnica de *transfer learning*.
```
base_model.trainable = False
```
As camadas abaixo são adicionadas ao final do modelo base. São elas que serão treinadas e sofrerão alterações em seus pesos. Utilizei camadas recomendadas para classificação de imagem, mas experimentar com esse segmento do modelo pode ser frutífero.
```
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)
```

Nas linhas seguintes acontece a definição do modelo final, passando pela camada de pré-processamento e modelo base importados.
```
inputs = tf.keras.Input(shape = (224, 224, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
#x = tf.keras.layers.Dropout(0.2)(x) 
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)
```
O objetivo da camada de **dropout** é evitar *overfitting*, ignorando alguns nós de maneira aleatória. A adição dessa camada, porém, não gerou impacto significante nos resultados (e ela teoricamente reduz a velocidade de evolução do treinamento) e não houve *overfitting* aparente nos testes que eu fiz sem **dropout**. Deixei ela comentada para possível uso futuro.

Na compilação do modelo, foi escolhida a função de *loss* **BinaryCrossentropy**, próprio para classificação binária. Foi utilizada a métrica **accuracy**, mas cheguei a fazer testes utilizando a métrica **AUC** (*Area Under the Curve*) e houve leve aumento na velocidade de evolução do treinamento.

```
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
    metrics=['accuracy']) 
```

Após o treinamento, o script salva tanto o estado final do modelo num arquivo de formato .h5 quanto o histórico das métricas e a duração de treinamento em pastas de cada modelo em [history](MainTraining/history).

O script [Test.py](MainTraining/Test.py) pode ser usado para testar os modelos.

### Pasta FederatedLearning

Nessa pasta, lidamos com o aprendizado federado através do framework Flower. Existem 2 scripts principais, [Client.py](FederatedLearning/Client.py) e [Server.py](FederatedLearning/Server.py), que atuam em papel de cliente e servidor.

#### Servidor

No script do servidor, é definida a estratégia de aprendizado (nesse caso, FedAvg), podendo escolher parâmetros como número mínimo de clientes conectados, definir os pesos iniciais do modelo, etc.

```
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5
)
```
O **while loop** seguinte executa uma vez para cada rede neural pré-treinada a ser utilizada. Nele, é iniciado o **servidor Flower**, com algumas configurações como o ip utilizado e o número de rounds de treinamento (Ao final de cada round, os clientes se comunicam com o servidor e ocorre a agregação dos pesos). O tempo gasto no treinamento de cada modelo é, então, salvo no arquivo [time.txt](FederatedLearning/history/time.txt).

#### Cliente

O script do cliente, ocorre a importação dos dados da mesma forma que foi realizada anteriormente, mas desta vez, dentro da pasta [traindata](FederatedLearning/traindata), há uma pasta para cada cliente.

Em seguida, é definido o **cliente Flower**, com as funções que o servidor solicitará a execução em diferentes momentos durante os rounds de treinamento. A função **get_parameters()** é chamada quando o servidor deseja saber os pesos do cliente e neste caso apenas retornamos os pesos do modelo. A função **fit()** é executada durante o treinamento em si e nela também escolhi salvar o histórico das métricas a cada época em arquivos txt na pasta [history](FederatedLearning/history). A função **evaluate()** é chamada na hora de avaliar o modelo de cada cliente e obter as métricas.

O **while loop** seguinte funciona de maneira análoga ao do script [Classification.py](MainTraining/Classification.py), definindo o modelo base e as camadas adicionais, escolhendo a função de *loss* e iniciando o **cliente Flower** no final. Note que o próximo ciclo do *loop* só inicia após o término do treinamento do ciclo atual.

#### Script Run.py

Esse script é destinado para a execução automática da simulação, executando primeiro o script do servidor e depois algumas instâncias do script do cliente, alterando o parâmetro **client_id**.

> [!NOTE]
> Esse script não está funcionando da maneira desejada, os processos não estão sendo executados em paralelo e são executados apenas quando o anterior termina a execução.

O jeito que eu estava fazendo antes era criar uma cópia de cada script e mudar o parâmetro client_id manualmente, e então eu executava cada um deles separadamente.

## Passos futuros/a fazer

- Realizar uma comparação mais profunda entre as diferentes redes neurais utilizadas sob a luz de mais aspectos.
  - Matriz de confusão.
  - Utilizar mais métricas.
    
- Explorar melhor as opções do Flower.
  - Fazer testes e comparações com outras estratégias de aprendizado além do FedAvg.
  - Experimentar com diferentes parâmetros das estratégias.

- Utilizar validação cruzada.
  - Reduzir o impacto da aleatoriedade da divisão dos conjuntos de treino e validação.

- Utilizar um conjunto de dados de imagens capturadas nas vias brasileiras, com o intuito de se adequar melhor às peculiaridades do nosso ambiente, como condições de luminosidade, tipo de asfalto, tipo de vegetação, cor do solo, etc.

- Realização de testes de treinamento em dispositivos móveis, simulando melhor uma situação real do aprendizado federado, com poder de processamento menor e separação física entre os clientes, fazendo com que o atraso da rede e a possível perda de pacotes seja levada em conta.

