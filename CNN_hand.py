#data: train and val
#from keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
import tensorflow.compat.v1 as tf
import cv2
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
#import wget

#split into train and test
tfds.disable_progress_bar()
tf.enable_v2_behavior()

#load MNIST
(ds_train, ds_test), ds_info = tfds.load(
    'rock_paper_scissors',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

training_datagen = ImageDataGenerator(
      rescale = 1./255,
	  rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

#print(ds_train,ds_test,ds_info)

image, label = tfds.as_numpy(tfds.load(
    'rock_paper_scissors',
    split='test',
    batch_size=-1,
    as_supervised=True,
))

#print(type(image), image.shape)


#build training pipeline
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

#evaluation pipeline
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


#create and train model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(330, 330, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(input_shape=(300, 300, 3)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

'''
model = tf.keras.models.Sequential([ 
     model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(300, 300, 3))),  
     tf.keras.layers.Flatten(),                      
     tf.keras.layers.Dense(128,activation='relu'),
     tf.keras.layers.Dense(10, activation='softmax')                    
])
'''

#optimize
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

model.fit(
    ds_train,
    epochs=5,
    validation_data=ds_test,
)

# Display the model's architecture
model.summary()


# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('my_model.h5')

def resizeImage():
    return cv2.resize(pth, (300, 300)).reshape(1, 300, 300, 3)

def main():
    pass

    for i in range(90):
        ret, frame = cap.read()
        my_image = resizeImage(frame)
        predictions = model.predict(my_image)

        # Countdown
        if i // 20 < 3:
            frame = cv2.putText(frame, str(i // 20 + 1), (320, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (250, 250, 0), 2,
                                cv2.LINE_AA)

        elif i / 20 < 3.5:
            pred = arr_to_shape[np.argmax(loaded_model.predict(prepImg(frame[50:350, 100:400])))]


#https://towardsdatascience.com/building-a-rock-paper-scissors-ai-using-tensorflow-and-opencv-d5fc44fc8222
#https://www.tensorflow.org/datasets/catalog/rock_paper_scissors
#https://colab.research.google.com/github/tensorflow/datasets/blob/master/docs/keras_example.ipynb#scrollTo=J8y9ZkLXmAZc
