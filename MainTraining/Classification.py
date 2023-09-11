import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import MobileNetV3Large

import numpy as np
import time


# Divisão dos datasets de treinamento e validação (utilizando função do keras)
image_size = (224, 224)
image_shape = image_size + (3,)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Potholes classification/traindata',
    validation_split = 0.2,
    subset = 'training',
    seed = 47,
    image_size = image_size,
    batch_size = batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Potholes classification/traindata',
    validation_split = 0.2,
    subset = 'validation',
    seed = 47,
    image_size = image_size,
    batch_size = batch_size
)

train_ds = train_ds.prefetch(buffer_size = 32)
val_ds = val_ds.prefetch(buffer_size = 32)



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
    x = tf.keras.layers.Dropout(0.2)(x) # Adicionando uma camada de dropout no meio para evitar overfitting
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs)


    # Compila o modelo

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                metrics=['accuracy']) 

    # Treinamento
    start_time = time.time()
    H = model.fit(train_ds,
                epochs=100,
                validation_data=train_ds)

    # Salva o estado atual do modelo
    print("Salvando modelo...")
    model.save('save' + names[rep] + '.h5', save_format="h5")

    # Salva o histórico das métricas escolhidas em arquivos txt 
    with open('MainTraining/history/' + names[rep] + '/accuracy' + names[rep] + '.txt', 'a') as f:
        for i in H.history['accuracy']:
            f.writelines(str(i) + '\n')
    with open('MainTraining/history/' + names[rep] + '/val_accuracy' + names[rep] + '.txt', 'a') as f:
        for i in H.history['val_accuracy']:
            f.writelines(str(i) + '\n')
    with open('MainTraining/history/' + names[rep] + '/loss' + names[rep] + '.txt', 'a') as f:
        for i in H.history['loss']:
            f.writelines(str(i) + '\n')
    with open('MainTraining/history/' + names[rep] + '/val_loss' + names[rep] + '.txt', 'a') as f:
        for i in H.history['val_loss']:
            f.writelines(str(i) + '\n')
    with open('MainTraining/history/time.txt', 'a') as f:
        f.writelines(names[rep] + ' - ' + str(time.time() - start_time) + '\n')

    rep += 1

# Para testar algum modelo, utilizar o método a seguir (sendo "val_ds" o dataset a ser testado):

#testModel = tf.keras.models.load_model("save100epochs.h5")
#results = testModel.evaluate(test_ds, batch_size = 32)

#print("test loss, test acc:", results)

