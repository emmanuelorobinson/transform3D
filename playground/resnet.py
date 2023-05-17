import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

# tf.keras.applications.resnet50.ResNet50
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

# from tensorflow.python.keras.models import Sequential old version
from tensorflow.keras.models import Sequential

# from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dense

# from tensorflow.python.keras import optimizers
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#tf metrics
from tensorflow.keras.metrics import Precision, Recall


from PIL import Image

# Fixed for our Cats & Dogs classes
NUM_CLASSES = 13

# Fixed for Cats & Dogs color images
CHANNELS = 3

IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = "avg"
DENSE_LAYER_ACTIVATION = "softmax"
OBJECTIVE_FUNCTION = "categorical_crossentropy"

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ["accuracy", Precision(), Recall()]

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 15

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 64
BATCH_SIZE_VALIDATION = 64


# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1

# resnet_weights_path = 'resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# resnet_weights_path = ResNet50
# Still not talking about our train/test data or any pre-processing.


import datetime
now = datetime.datetime.now()
date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

model = Sequential()

# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
model.add(
    ResNet50(include_top=False, pooling=RESNET50_POOLING_AVERAGE, weights="imagenet")
)

# 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
model.add(Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION))

# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = False

# Compile the model
sgd = optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=OBJECTIVE_FUNCTION, metrics=LOSS_METRICS)

image_size = IMAGE_RESIZE

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

#  https://keras.io/preprocessing/image/
train_generator = train_datagen.flow_from_directory(
    directory=r"./dataset/train/",
    target_size=(image_size, image_size),
    color_mode="rgb",
    class_mode="categorical",
    shuffle=True,
    batch_size=BATCH_SIZE_VALIDATION,
)

valid_generator = train_datagen.flow_from_directory(
    directory=r"./dataset/valid/",
    target_size=(image_size, image_size),
    color_mode="rgb",
    class_mode="categorical",
    shuffle=True,
    batch_size=BATCH_SIZE_VALIDATION,
)

TRAINING_SIZE = train_generator.n
VALIDATION_SIZE = valid_generator.n

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = TRAINING_SIZE // BATCH_SIZE_TRAINING
STEPS_PER_EPOCH_VALIDATION = VALIDATION_SIZE // BATCH_SIZE_VALIDATION



(BATCH_SIZE_TRAINING, len(train_generator), BATCH_SIZE_VALIDATION, len(valid_generator))

cb_early_stopper = EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(
    # filepath="./ImgClassModelBest.h5",
    filepath="./runs/ImgClassModelBest_" + date_time + ".h5",
    monitor="val_loss",
    save_best_only=True,
    mode="auto",
)

fit_history = model.fit_generator(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
    epochs=NUM_EPOCHS,
    validation_data=valid_generator,
    validation_steps=STEPS_PER_EPOCH_VALIDATION,
    callbacks=[cb_checkpointer, cb_early_stopper]
    # callbacks=[cb_checkpointer]
)


# model.save_weights("./best.hdf5")
# model.save("./ImgClassModelV2.h5")
model.save("./runs/ImgClassModelV2_" + date_time + ".h5")



# add date and time to file name


# save epoch, loss, accuracy, val_loss, val_accuracy to csv
df = pd.DataFrame(fit_history.history)
df.to_csv("./runs/history_" + date_time + ".csv", index=False)
# df.to_csv("./history.csv", index=False)


print(fit_history.history.keys())


plt.figure(1, figsize=(15, 8))

plt.subplot(221)
plt.plot(fit_history.history["accuracy"])
plt.plot(fit_history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "valid"])

plt.subplot(222)
plt.plot(fit_history.history["loss"])
plt.plot(fit_history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "valid"])

plt.show()

test_generator = train_datagen.flow_from_directory(
    directory=r"./evaluate/",
    target_size=(image_size, image_size),
    color_mode="rgb",
    batch_size=BATCH_SIZE_TESTING,
    class_mode=None,
    shuffle=False,
    seed=123,
)

test_generator.reset()

pred = model.predict_generator(test_generator, steps=len(test_generator), verbose=1)

predicted_class_indices = np.argmax(pred, axis=1)

ROOT = os.path.dirname(os.getcwd())
TEST_DIR = ROOT + "/3D_Printed_Materials_AI/evaluate/"


num_cols = 7
num_rows = 7

f, ax = plt.subplots(num_rows, num_cols, figsize=(15, 15))


# get nubmer of images in test set
num_test_images = len(test_generator.filenames)
for i in range(num_test_images):
    imgBGR = cv2.imread(TEST_DIR + test_generator.filenames[i])
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

    # train_generator.class_indices.items()
    predicted_class = list(train_generator.class_indices.keys())[
        list(train_generator.class_indices.values()).index(predicted_class_indices[i])
    ]

    row = i // num_cols
    col = i % num_cols

    ax[row, col].imshow(imgRGB)
    ax[row, col].axis("off")
    ax[row, col].set_title("Predicted:{}".format(predicted_class))


plt.show()


# class_names = {
#     0: "framestl",
#     1: "frontstl",
#     2: "lockblock",
#     3: "magazine_catch",
#     4: "rearsight",
#     5: "sidecover",
#     6: "stoplever",
#     7: "trigger",
#     8: "trigger_bar",
#     9: "triggermech",

# }

# # load the best model
# model = tf.keras.models.load_model("./ImgClassModel.h5")

# # predict an image from test set
# img = os.listdir(TEST_DIR + '/test/')[0]
# print("Image:", img)

# # expected shape is (None, None, 3)
# # x = image.load_img(TEST_DIR + '/test/' + img, target_size=(224, 224))
# x = tf.keras.preprocessing.image.load_img(TEST_DIR + '/test/' + img, target_size=(224, 224))

# # x = image.img_to_array(x)
# x = tf.keras.preprocessing.image.img_to_array(x)

# x = np.expand_dims(x, axis=0)

# x = preprocess_input(x)

# preds = model.predict(x)


# # print the predicted class name and the probability
# class_index = preds.argmax(axis=-1)

# # sort class_names alaphabetically
# class_names = {k: v for k, v in sorted(class_names.items(), key=lambda item: item[1])}

# #use class_index to get the class name
# classes = list(class_names.values())[list(class_names.keys()).index(class_index[0])]

# # get preodiction probability
# prob = preds[0][class_index][0]

# # convert probability to percentage and add % sign
# prob = round(prob * 100, 2)

# print("Predicted class:", classes)
# print(f'Probability: {prob}%')




# ??train_datagen.flow_from_directory
