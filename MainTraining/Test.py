import tensorflow as tf

image_size = (224, 224)
image_shape = image_size + (3,)
batch_size = 32

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Potholes classification/testdata',
    validation_split = 1.0,
    subset = 'validation',
    seed = 47,
    image_size = image_size,
    batch_size = batch_size
)

testModel = tf.keras.models.load_model("saveVGG.h5")
results = testModel.evaluate(test_ds, batch_size = 32)
print("test loss, test acc:", results)