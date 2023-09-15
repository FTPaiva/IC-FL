import sys
import flwr as fl
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import MobileNetV3Large

client_id = sys.argv[1]
client_id = str(client_id)

print('client ' + client_id + ' initialized')


# Divisão dos datasets de treinamento e validação (utilizando função do keras)
image_size = (224, 224)
image_shape = image_size + (3,)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'traindata/' + client_id,
    validation_split = 0.2,
    subset = 'training',
    seed = 47,
    image_size = image_size,
    batch_size = batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'traindata/' + client_id,
    validation_split = 0.2,
    subset = 'validation',
    seed = 47,
    image_size = image_size,
    batch_size = batch_size
)

train_ds = train_ds.prefetch(buffer_size = 32)
val_ds = val_ds.prefetch(buffer_size = 32)


# Define o cliente Flower
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        H = model.fit(train_ds, epochs=5, batch_size=32, validation_data=val_ds)

        # Registra o histórico de métricas de cada round de treinamento

        with open('FederatedLearning/history/' + names[rep] + '/accuracy' + client_id + names[rep] + '.txt', 'a') as f:
            for i in H.history['accuracy']:
                f.writelines(str(i) + '\n')
        with open('FederatedLearning/history/' + names[rep] + '/val_accuracy' + client_id + names[rep] + '.txt', 'a') as f:
            for i in H.history['val_accuracy']:
                f.writelines(str(i) + '\n')
        with open('FederatedLearning/history/' + names[rep] + '/loss' + client_id + names[rep] + '.txt', 'a') as f:
            for i in H.history['loss']:
                f.writelines(str(i) + '\n')
        with open('FederatedLearning/history/' + names[rep] + '/val_loss' + client_id + names[rep] + '.txt', 'a') as f:
            for i in H.history['val_loss']:
                f.writelines(str(i) + '\n')

        return model.get_weights(), len(train_ds), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy, *rest = model.evaluate(val_ds, batch_size = 32)
        print("loss = " + str(loss) + ", acc = " + str(accuracy))
        return loss, len(val_ds), {"accuracy": accuracy}

# Lista para nomes de arquivos
names = ['VGG', 'ResNet', 'Xception', 'MobileNet']

rep = 0
while (rep <= 3): # Executa uma vez para cada modelo a ser testado

    # Importação do modelo base
    # Utiliza função de preprocessamento específico de cada modelo, fornecidas pelo keras

    if (rep == 0):
        base_model = VGG16(input_shape=image_shape,
                            include_top=False,
                            weights='imagenet')
        preprocess_input = tf.keras.applications.vgg16.preprocess_input 

    elif (rep == 1):
        base_model = ResNet50V2(input_shape=image_shape,
                            include_top=False,
                            weights='imagenet')
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

    elif (rep == 2):
        base_model = Xception(input_shape=image_shape,
                            include_top=False,
                            weights='imagenet')
        preprocess_input = tf.keras.applications.xception.preprocess_input

    elif (rep == 3):
        base_model = MobileNetV3Large(input_shape=image_shape,
                            include_top=False,
                            weights='imagenet') 
        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

    base_model.trainable = False # Impede que as camadas do modelo base de sejam alteradas

    # Classification head, são as camadas que serão alteradas durante o treinamento

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = tf.keras.layers.Dense(1)


    # Modelo final

    inputs = tf.keras.Input(shape = (224, 224, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    #x = tf.keras.layers.Dropout(0.2)(x) # Adicionando uma camada de dropout para evitar overfitting
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)


    # Compila o modelo

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # Inicia o cliente Flower
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient())
    
    rep += 1