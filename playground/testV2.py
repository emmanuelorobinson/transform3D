import datetime
import os
import shutil

import numpy as np
import tensorflow as tf

import albumentations as A
import matplotlib.pyplot as plt
import vtk as vtk
import vtkplotlib as vpl

from tensorflow.keras.applications.resnet50 import preprocess_input

from stl import mesh
from stl.mesh import Mesh
from PIL import Image



# set root directory
ROOT = os.path.dirname(os.path.abspath(__file__))
DIRECTORY = "./evaluate/"
MESH_DIRECTORY = "./mesh_eval/"
# set camera resolution
RESOLUTION_X = 1024
RESOLUTION_Y = 1024

# camera rotation
ROTATION = [
    (90, 0, 0),
    (270, 0, 0),
    (0, 90, 0),
    (0, 270, 0),
    (0, 0, 0),
    (0, 0, 180),
    (45, 45, 0),
    (45, 0, 45),
    (0, 45, 45),
    (30, 45, 30),
]


CLASSES = {
    0: "barrel",
    1: "bulletshell",
    2: "frame",
    3: "frontsight",
    4: "grip",
    5: "heel",
    6: "lockblock",
    7: "magazine",
    8: "rearsight",
    9: "slide",
    10: "stock",
    11: "stoplever",
    12: "trigger",
}


# take picture in 6 directions and save to file
def take_picture(mesh, mesh_name):

    fig = vpl.figure()

    # load mesh
    vpl.mesh_plot(mesh_data=mesh, color="#696969")

    for i in range(len(ROTATION)):
        # reset camera on each iteratin
        vpl.reset_camera()

        # rotate camera
        vpl.view(camera_direction=ROTATION[i])

        # check if DIRECTORY exists and create if not
        if os.path.isdir(DIRECTORY) == False:
            os.mkdir(DIRECTORY)


        vpl.save_fig(
            path=DIRECTORY + mesh_name + str(i) + ".png",
            pixels=(1024, 1024),
            off_screen=True,
            magnification=10,
            fig=fig,
        )
        
        # t = Thread(target=vpl.save_fig, args=(DIRECTORY + mesh_name + str(i) + ".png", 1, (1024, 1024), 10, False, fig))
        # t.start()
        # t.join()

        vpl.reset_camera()

        print("done saving picture " + str(i) + " of " + mesh_name)

    # clear mesh
    vpl.close()

    # return 6 images


def read_mesh(file):
    mesh_name = file
    mesh_name = mesh_name.replace(".stl", "")
    # check if there is a space in the mesh name
    if " " in mesh_name:
        # replace with underscore
        mesh_name = mesh_name.replace(" ", "_")

    # lower case
    mesh_name = mesh_name.lower()
    # get mesh file
    mesh_file = MESH_DIRECTORY + file
    # load mesh
    mesh = Mesh.from_file(mesh_file)

    return mesh, mesh_name

    # take_picture(mesh, mesh_name)
    
    # call take_picture in main thread


def predict_img(image):
    # load the best model
    model = tf.keras.models.load_model("./ImgClassModelV2.h5")

    # predict an image from test set
    # print(TEST_DIR + "test/")
    img = "./evaluate/" + image

    # create directory for test images

    # img = f"./evaluate/{image}"

    # expected shape is (None, None, 3)
    # x = image.load_img(TEST_DIR + '/test/' + img, target_size=(224, 224))
    # x = tf.keras.preprocessing.image.load_img(TEST_DIR + '/test/' + img, target_size=(224, 224))

    x = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))

    # x = image.img_to_array(x)
    x = tf.keras.preprocessing.image.img_to_array(x)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)

    # print the predicted class name and the probability
    class_index = preds.argmax(axis=-1)

    # sort class_names alaphabetically
    class_names = {k: v for k, v in sorted(CLASSES.items(), key=lambda item: item[1])}

    # use class_index to get the class name
    classes = list(class_names.values())[list(class_names.keys()).index(class_index[0])]

    # get preodiction probability
    prob = preds[0][class_index][0]

    # convert probability to percentage and add % sign
    prob = round(prob * 100, 2)

    # delete test directory
    # shutil.rmtree("test")

    return {"class": classes, "probability": prob}
    # print("Predicted class:", classes)
    # print(f'Probability: {prob}%')


def predict_stl():
    # for i in evalueate directory which contains the images

    evaluation = []

    for file in os.listdir(DIRECTORY):
        # check if file is an image
        if file.endswith(".png"):
            # predict image
            prediction = predict_img(file)
            # append prediction to evaluation list
            evaluation.append(prediction)
    # prediction = {"class": "framestl", "probability": 99.99}

    # for the number of different classes that are the same in the evaluation list
    # get the average of the probabilities and append to a new list
    # return all the classes and their average probabilities
    # return evaluation

    # get all the classes in the evaluation list
    classes = [i["class"] for i in evaluation]
    # get all the probabilities in the evaluation list
    probabilities = [i["probability"] for i in evaluation]

    # get the unique classes
    unique_classes = list(set(classes))

    # create a list to hold the average probabilities
    average_probabilities = []

    # for each unique class
    for i in unique_classes:
        # get the index of the class in the classes list
        index = [j for j, x in enumerate(classes) if x == i]
        # get the probabilities of the class
        prob = [probabilities[j] for j in index]
        # get the average of the probabilities
        average = sum(prob) / len(prob)
        # append the average to the average_probabilities list
        average_probabilities.append(average)

    # create a dictionary to hold the class and the average probability
    average_probabilities_dict = dict(zip(unique_classes, average_probabilities))

    # return the dictionary
    return average_probabilities_dict

def main():
    # for each file in the directory
    for file in os.listdir(MESH_DIRECTORY):
        # check if file is an stl
        if file.endswith(".stl"):
            # read mesh
            mesh, mesh_name = read_mesh(file)
            # take picture
            # take_picture(mesh, mesh_name)

    # predict stl
    prediction = predict_stl()

    # sort dictionary by value
    prediction = dict(sorted(prediction.items(), key=lambda item: item[1], reverse=True))

    # print the prediction
    print(prediction)


if __name__ == "__main__":
    main()
