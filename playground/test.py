import torch
import torchvision.models as models
import os
import random
import shutil
import math
import time

import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import vtk as vtk

import vtkplotlib as vpl
from stl import mesh
from stl.mesh import Mesh
from PIL import Image


#directory location

# root is one level above the current directory
# ROOT = os.path.dirname(os.getcwd())

# DIRECTORY= ROOT + '/evaluate/'
# MESH_DIRECTORY = ROOT + '/mesh_eval/'

DIRECTORY = './evaluate/'
MESH_DIRECTORY = './mesh_eval/'

# set camera resolution
RESOLUTION_X = 1024
RESOLUTION_Y = 1024

# camera rotation
ROTATION = [(90, 0, 0), (270, 0, 0), (0, 90, 0), (0, 270, 0), (0, 0, 0), (0, 0, 180),(45, 45, 0),(45, 0, 45),(0, 45, 45),(30, 45, 30)]


# names:
#   0: frontsl
#   1: rearsight
#   2: slidecover
#   3: framestl
#   4: triggermech
#   5: lockblock
#   6: stoplever
#   7: trigger
#   8: magazine_catch
#   9: background


# file = os.listdir(MESH_DIRECTORY)
# mesh_name = file.split('_')[1]
# mesh_name = mesh_name.replace('.stl', '')
# # check if there is a space in the mesh name
# if ' ' in mesh_name:
# # replace with underscore
#   mesh_name = mesh_name.replace(' ', '_')
     
# # lower case
# mesh_name = mesh_name.lower()
     
# # get mesh file
# mesh_file = MESH_DIRECTORY + file
# # load mesh
# mesh = Mesh.from_file(mesh_file)



# take picture in 6 directions and save to file
def take_picture(mesh, mesh_name):
    
    fig = vpl.figure()
    
    # load mesh
    vpl.mesh_plot(mesh_data=mesh, color='#696969')

    # rotate figure to get 6 images
    
      

    # length of rotation


    for i in range(len(ROTATION)):
        # reset camera on each iteratin
        vpl.reset_camera()

        # rotate camera
        vpl.view(camera_direction=ROTATION[i])

        # check if DIRECTORY exists and create if not
        if (os.path.isdir(DIRECTORY) == False):
            os.mkdir(DIRECTORY)

        # save image
        vpl.save_fig(path=DIRECTORY + mesh_name + str(i) + '.png', pixels=(1024, 1024), off_screen=True, magnification=10, fig=fig)
        vpl.reset_camera()

    # clear mesh
    vpl.close()

    # return 6 images


def get_class_name(num):
  if num == 0:
    return 'frontsl'
  elif num == 1:
    return 'rearsight'
  elif num == 2:
    return 'slidecover'
  elif num == 3:
    return 'framestl'
  elif num == 4:
    return 'triggermech'
  elif num == 5:
    return 'lockblock'
  elif num == 6:
    return 'stoplever'
  elif num == 7:
    return 'trigger'
  elif num == 8:
    return 'magazine_catch'
  elif num == 9:
    return 'background'
  else:
    return 'nil'
  

    

def evaluate():
  # go to evaluate directory containing all the images
  # run all the images through the model, get the average prediction of the 6 images
  # return the average prediction

  model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

  # get all the images in the directory
  imgs = os.listdir(DIRECTORY)

  # create a list to store the predictions
  predictions = {
    'nil' : {
      'confidence' : 0,
      'length' : 0,
      'avg_confidence' : 0,
    }
  }

  result = 0
  # loop through all the images
  for img in imgs:
    
    # print(img)

    pred = model(DIRECTORY + img)
    # print(pred)

    # get dimension and size of pred
    dimension = pred.xyxy[0].shape
    size = dimension[0]
  
    if (size == 0):
      # no prediction
      continue
    else:
    # get confidence of prediction
      confidence = pred.xyxy[0][0][4].item()
    



    # get class name
    class_name = pred.xyxy[0][0][5].item()
    class_name = int(class_name) - 1

    # if key does not exist in predictions, add it
    if class_name not in predictions:
      predictions[class_name] = {
        'confidence' : 0,
        'length' : 0,
        'avg_confidence' : 0,
      }
    
    
    predictions[class_name]['confidence'] += confidence
    predictions[class_name]['length'] += 1

    predictions['nil']['confidence'] += 1 - confidence
    predictions['nil']['length'] += 1


  
  # get average confidence and print along with class name
  
  for key in predictions:
    
    # get average confidence
    predictions[key]['avg_confidence'] = predictions[key]['confidence'] / predictions[key]['length']

    # print class name and average confidence
    print(get_class_name(key), predictions[key]['avg_confidence'])
     


def clear_cache():
  # clear evaluate directory

  # check if DIRECTORY exists and create if not
  if (os.path.isdir(DIRECTORY) == False):
    # noting to clear
    return

  # remove all files in DIRECTORY
  files = os.listdir(DIRECTORY)
  for file in files:
    file_path = DIRECTORY + file
    os.remove(file_path)

  # remove DIRECTORY
  os.rmdir(DIRECTORY)


for file in os.listdir(MESH_DIRECTORY):
  mesh_name = file.split('_')[1]
  mesh_name = mesh_name.replace('.stl', '')
  # check if there is a space in the mesh name
  if ' ' in mesh_name:
  # replace with underscore
    mesh_name = mesh_name.replace(' ', '_')
     
  # lower case
  mesh_name = mesh_name.lower()
     
  # get mesh file
  mesh_file = MESH_DIRECTORY + file
  # load mesh
  mesh = Mesh.from_file(mesh_file)

  take_picture(mesh, mesh_name)
# result = evaluate()
# clear_cache()
# print(result)
     
     



