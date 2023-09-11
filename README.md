# IC-FedLearn

## Base de dados utilizada

A base de dados escolhida foi a [Road Pothole Images for Pothole detection](https://www.kaggle.com/datasets/sovitrath/road-pothole-images-for-pothole-detection/) disponível no Kaggle.
Ela possui 3777 imagens, sendo 1119 imagens com buracos e 2658 sem buracos, todas captadas por uma câmera veicular,
no estilo *dashcam* que é um modelo de câmera que acopla-se ao painel do automóvel, registrando a visão do motorista. Vale notar que todas
as imagens foram capturadas na África do Sul e possuem características específicas da região, como coloração da terra, tipo de vegetação,
condições de iluminação, tipo de calçada, tamanho e abundância de prédios, etc.

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
O script principal ([Classification.py](MainTraining/Classification.py)) usa os dados contidos na pasta [traindata](MainTraining/traindata), onde as imagens classificadas como **positivas** (contendo buracos) ficam em uma pasta e as **negativas** em outra.

A divisão dos datasets de treinamento e validação é feita através
de uma função do keras (*tf.keras.preprocessing.image_dataset_from_directory*). Nos parâmetros dessa função, defini que 20% das imagens seriam destinadas à validação e o batch size seria de 32 (esses valores pode ser alterados a fim de otimização, mas
utilizei valores que vi sendo recomendados). O tamanho de imagem de 224x224 foi utilizado pois a maioria das redes neurais pré-treinadas que usei tinham esse formato de input. Foi utilizada uma seed fixa para que os resultados sejam consistentes
e as comparações feitas entre os modelos sejam mais justas.

Há um while loop que executa uma vez para cada rede a ser treinada, mudando partes do modelo a cada loop. Primeiramente, é definido qual rede pré-treinada deve ser importada, juntamente com sua função de pré-processamento específica (que prepara as
imagens para se adequarem aos inputs esperados pela rede). É importante definir o parâmetro *include_top* da rede como **false**, pois a fim de realizar o transfer learning, alteramos o final da rede neural, então não desejamos importar as últimas camadas.

O trecho abaixo impede que as camadas do modelo base de sejam alteradas, essencial para a técnica de transfer learning.
```
base_model.trainable = False
```

dasdasasdasdasdadasd

## Passos futuros/a fazer

- Realizar uma comparação mais profunda entre as diferentes redes neurais utilizadas sob a luz de mais aspectos.
  - Matriz de confusão.
  - Utilizar mais métricas.

- Utilizar validação cruzada.
  - Reduzir o impacto da aleatoriedade da divisão dos conjuntos de treino e validação.

- Utilizar um conjunto de dados de imagens capturadas nas vias brasileiras, com o intuito de se adequar melhor às peculiaridades do nosso ambiente,
como condições de luminosidade, tipo de asfalto, tipo de vegetação, cor do solo, etc.

- Realização de testes de treinamento em dispositivos móveis, simulando melhor uma situação
real do aprendizado federado, com poder de processamento menor e separação física entre os clientes, fazendo com que o atraso da
rede e a possível perda de pacotes seja levada em conta.

