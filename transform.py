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

# set number of training data

TRAINING_PERCENTAGE = 0.8

#directory location
DIRECTORY= os.getcwd() + '/dataset/'
BACKGROUND_PATH = os.getcwd() + '/background/'
MESH_DIRECTORY = os.getcwd() + '/mesh/'

# set camera resolution
RESOLUTION_X = 1024
RESOLUTION_Y = 1024

# camera rotation
ROTATION = [(90, 0, 0), (270, 0, 0), (0, 90, 0), (0, 270, 0), (0, 0, 0), (0, 0, 180),(45, 45, 0),(45, 0, 45),(0, 45, 45),(30, 45, 30)]


BACKGROUND_FILES = os.listdir(BACKGROUND_PATH)

#A.augmentations.blur.transforms.GlassBlur (sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=1)
TRANSFORM_LIST=[A.augmentations.geometric.transforms.VerticalFlip(p=1),
                A.augmentations.geometric.transforms.Transpose(p=1),
                A.augmentations.geometric.transforms.ShiftScaleRotate (shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, rotate_method='largest_box', always_apply=False, p=1),
                A.augmentations.geometric.transforms.PiecewiseAffine (scale=(0.03, 0.05), nb_rows=4, nb_cols=4, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, always_apply=False, keypoints_threshold=0.01, p=1),
                A.augmentations.geometric.resize.SmallestMaxSize (max_size=1024, interpolation=1, always_apply=False, p=1),
                A.augmentations.geometric.rotate.SafeRotate (limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=1),
                A.augmentations.geometric.transforms.HorizontalFlip(p=1),
                
                A.augmentations.transforms.RandomContrast (limit=0.2, always_apply=False, p=1),
                A.augmentations.transforms.RGBShift (r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=1),
                A.RandomBrightnessContrast(p=0.2)
]


# take picture in 6 directions and save to file
def take_picture(mesh, mesh_name):
    
    # load mesh
    vpl.mesh_plot(mesh_data=mesh, color='#696969')

    for i in range(len(ROTATION)):
        # reset camera on each iteratin
        vpl.reset_camera()

        # rotate camera
        vpl.view(camera_direction=ROTATION[i])

        # check if DIRECTORY exists and create if not
        if (os.path.isdir(DIRECTORY) == False):
            os.mkdir(DIRECTORY)

        # save image
        vpl.save_fig(path=DIRECTORY + mesh_name + str(i) + '.png', pixels=(1024, 1024), off_screen=True)
        vpl.reset_camera()

    # clear mesh
    vpl.close()

    # return 6 images

# convert bounding box to yolo format
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

# find bounding box of object in image
def camera_view_bounds_2d(file):
  # load image using cv2
  image = cv2.imread(file)

  # convert to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # threshold image
  thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  # find contours
  contours = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]

  # draw bounding box
  return_x, return_y, return_w, return_h = 0, 0, 0, 0

  for i in contours:
    # get bounding box
    x,y,w,h = cv2.boundingRect(i)
    cv2.rectangle(image,(x,y),(x+w,y+h),(225,0,0),4)

    xmin = x
    xmax = x + w
    ymin = y
    ymax = y + h

    # convert to yolo format
    return_x, return_y, return_w, return_h = convert((RESOLUTION_X, RESOLUTION_Y), (xmin, xmax, ymin, ymax))
  
  # cv2.imwrite('image.png', image)
  
  return return_x, return_y, return_w, return_h

# create yolo label file
def create_yolo_label(label, mesh_name):
  for i in range(len(ROTATION)):
    # create file
    file_name = DIRECTORY + mesh_name + str(i) + '.txt'

    # write to file
    with open(file_name, "w+") as file:
      # write label
      x, y, w, h = camera_view_bounds_2d(DIRECTORY + mesh_name + str(i) + '.png')

      # write to file
      file.write(str(label) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h))

# augment images   
def augment(label, mesh_name):
  labels=['one']
  i=0

  # iterate through all images in directory
  for file in os.listdir(DIRECTORY):
    # check if file is an image
    if file.endswith(".png"):
        
        # load image
        image = cv2.imread(os.path.join(DIRECTORY, file))

        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load bounding box
        textfile=os.path.splitext(file)[0]+".txt"
        fl = open(os.path.join(DIRECTORY, textfile) ,'r')

        # read bounding box
        data = fl.readlines()
        fl.close()

        # convert bounding box to yolo format
        bbox = list(map(float, data[0].split(' ')))

        # convert to list
        bbox=bbox[1:]

        # add label
        bbox=bbox+["one"]

        # iterate through all transformations
        for newparam in TRANSFORM_LIST:
          
            # apply transformation
            newname="transform_"+ mesh_name +str(i)
            transform = A.Compose([newparam], bbox_params=A.BboxParams(format='yolo'))
            transformed = transform(image=image, bboxes=[bbox], class_labels=[labels])

            # save image
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']

            # save bounding box
            transformed_class_labels = transformed['class_labels']
            x, y, w, h,_ = transformed_bboxes[0]
            newbboxes=[label]+ [' '] +[str(x)]+ [' '] +[str(y)]+ [' '] +[str(w)]+ [' '] +[str(h)]
            s=''.join(newbboxes)
            with open(DIRECTORY+newname+".txt", "w") as output:
                output.write(s)
                output.close
            cv2.imwrite(DIRECTORY+newname+".jpg",transformed_image)
            i=i+1


# augment images   
def augment_no_box(label, mesh_name):
  labels=['one']
  i=0

  # iterate through all images in directory
  for file in os.listdir(DIRECTORY):
    # check if file is an image
    if file.endswith(".png"):
        
        # load image
        image = cv2.imread(os.path.join(DIRECTORY, file))

        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # iterate through all transformations
        for newparam in TRANSFORM_LIST:
          
            # apply transformation
            newname="_transform_"+ mesh_name +str(i)
            transform = A.Compose([newparam], bbox_params=A.BboxParams(format='yolo'))
            transformed = transform(image=image, bboxes=[])

            # save image
            transformed_image = transformed['image']

            
            cv2.imwrite(DIRECTORY+label+newname+".jpg",transformed_image)
            i=i+1


# remove background from image
def remove_bg(file):


   # Read image
  img = cv2.imread(file)
  hh, ww = img.shape[:2]

  # threshold on white
  # Define lower and uppper limits
  lower = np.array([200, 200, 200])
  upper = np.array([255, 255, 255])

  # Create mask to only select black
  thresh = cv2.inRange(img, lower, upper)

  # apply morphology
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
  morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

  # invert morp image
  mask = 255 - morph

  # apply mask to image
  result = cv2.bitwise_and(img, img, mask=mask)

  cv2.imwrite(file, result)

# remove background of a specific color from an image
# #d8dcd6
def remove_bg_color(file):

  # Read image
  image = Image.open(file, 'r')
  image = image.convert('RGBA')

  data = np.array(image)   # "data" is a height x width x 4 numpy array
  red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

  # Replace specific color with black... (leaves alpha values alone...)
  white_areas = (red == 216) & (green == 220) & (blue == 214)
  data[..., :-1][white_areas.T] = (0, 0, 0) # Transpose back needed

  image = Image.fromarray(data)

  # save image
  image.save(file)

  duplicate_images(file)

# add random background to image
def add_background(num, mesh_name):
  # load image using cv2
  imgFront  = cv2.imread(DIRECTORY + mesh_name + str(num) + '.png')

  # load background image
  imgBack = cv2.imread(BACKGROUND_PATH + random.choice(BACKGROUND_FILES))

  # resize front image
  height, width = imgFront.shape[:2]

  # resize background image
  resizeBack = cv2.resize(imgBack, (width, height), interpolation = cv2.INTER_CUBIC)

  # add background to front image
  for i in range(width):

    # check if pixel is black
    for j in range(height):
        
        pixel = imgFront[j, i]

        # if pixel is black, replace with background pixel
        if np.all(pixel == [0, 0, 0]):
            imgFront[j, i] = resizeBack[j, i] 

  # save image
  cv2.imwrite(DIRECTORY + mesh_name + str(num) + '.png', imgFront)

# add background to all images
def add_background_to_img(mesh_name):
   for i in range(len(ROTATION)):
    remove_bg_color(DIRECTORY + mesh_name + str(i) + '.png')
    add_background(i, mesh_name)
   


def duplicate_images(mesh_name):
  img_txt_name = mesh_name[:-4]
  shutil.copy(img_txt_name + '.png', img_txt_name + '_1.png')
  # shutil.copy(img_txt_name + '.txt', img_txt_name + '_1.txt')


def split_data():
  # function to split data into training and validation set

  train_path = DIRECTORY + 'train/'
  val_path = DIRECTORY + 'valid/'

  # get all files in directory
  files = os.listdir(DIRECTORY)

  # get all image files that ends with png or jpg
  image_files = [file for file in files if file.endswith(".png") or file.endswith(".jpg")]

  # get all text files
  text_files = [file for file in files if file.endswith(".txt")]

  # get number of files
  num_files = len(image_files)

  # get number of training files
  num_train = int(num_files * TRAINING_PERCENTAGE)

  # get number of validation files
  num_val = num_files - num_train

  # shuffle files
  random.shuffle(image_files)

  # get training files
  train_files = image_files[:num_train]

  # get validation files
  val_files = image_files[num_train:]

  # move training files to training directory
  for file in train_files:
    # create new file name train
    if (os.path.isdir(train_path) == False):
      os.mkdir(train_path)
    
    new_file_name = train_path + file[:-4]
    original_file_name = DIRECTORY + file[:-4]

    new_img_name = new_file_name + '.png'
    new_txt_name = new_file_name + '.txt'

    # check if jpg or png file
    if file.endswith(".jpg"):
      new_img_name = new_file_name + '.jpg'
      shutil.move(original_file_name + '.jpg', new_img_name)
    else:
       shutil.move(original_file_name + '.png', new_img_name)
       
      

    shutil.move(original_file_name + '.txt', new_txt_name)
    

  # move validation files to validation directory
  for file in val_files:
    # create new file name train
    if(os.path.isdir(val_path) == False):
      os.mkdir(val_path)

    new_file_name = val_path + file[:-4]
    original_file_name = DIRECTORY + file[:-4]

    new_img_name = new_file_name + '.png'
    new_txt_name = new_file_name + '.txt'

    # check if jpg or png file
    if file.endswith(".jpg"):
      new_img_name = new_file_name + '.jpg'
      shutil.move(original_file_name + '.jpg', new_img_name)
    else:
       shutil.move(original_file_name + '.png', new_img_name)
       
      
    shutil.move(original_file_name + '.txt', new_txt_name)

def get_class_name(num):
  if num == 0:
    return 'grip'
  elif num == 1:
    return 'frontsight'
  elif num == 2:
    return 'rearsight'
  elif num == 3:
    return 'slide'
  elif num == 4:
    return 'frame'
  elif num == 5:
    return 'barrel'
  elif num == 6:
    return 'lockblock'
  elif num == 7:
    return 'stoplever'
  elif num == 8:
    return 'trigger'
  elif num == 9:
    return 'magazine'
  elif num == 10:
    return 'stock'
  elif num == 11:
    return 'heel'
  elif num == 12:
    return 'bulletshell'
  


def split_data_img_class():
  # function to split data into training and validation set

  

  # get all files in directory
  files = os.listdir(DIRECTORY)

  # get all image files that ends with png or jpg
  image_files = [file for file in files if file.endswith(".png") or file.endswith(".jpg")]

  # get all text files
  text_files = [file for file in files if file.endswith(".txt")]

  # delete all text files
  for file in text_files:
    os.remove(DIRECTORY + file)


  

  # get number of files
  num_files = len(image_files)

  # get number of training files
  num_train = int(num_files * TRAINING_PERCENTAGE)

  # get number of validation files
  num_val = num_files - num_train

  # shuffle files
  random.shuffle(image_files)

  # get training files
  train_files = image_files[:num_train]

  # get validation files
  val_files = image_files[num_train:]

  # move training files to training directory
  for file in train_files:

    # number may be 1 or 2 digits

    # check if number is 2 digits 10_frontsight_1
    if file[1] == '_':
      class_num = int(file[0])
    else:
      class_num = int(file[0:2])

    # class_num = int(file[0])
    class_name = get_class_name(class_num)

    # train_path = DIRECTORY + class_name + '/train/'
    train_path = DIRECTORY + "train" + '/' + class_name + '/'
    
    # create new folder using train_path
    # check if directory exists
    if os.path.isdir(train_path) == False:
      os.makedirs(train_path)


    
    new_file_name = train_path + file[:-4]
    original_file_name = DIRECTORY + file[:-4]

    new_img_name = new_file_name + '.png'


    # check if jpg or png file
    if file.endswith(".jpg"):
      new_img_name = new_file_name + '.jpg'
      shutil.move(original_file_name + '.jpg', new_img_name)
    else:
       shutil.move(original_file_name + '.png', new_img_name)
       
      
    

  # move validation files to validation directory
  for file in val_files:

     # check if number is 2 digits 10_frontsight_1
    if file[1] == '_':
      class_num = int(file[0])
    else:
      class_num = int(file[0:2])
      
    class_name = get_class_name(class_num)

    # val_path = DIRECTORY + class_name + '/test/'
    val_path = DIRECTORY + "valid" + '/' + class_name + '/'

    # create new file name train
    if(os.path.isdir(val_path) == False):
      os.makedirs(val_path)

    new_file_name = val_path + file[:-4]
    original_file_name = DIRECTORY + file[:-4]

    new_img_name = new_file_name + '.png'
    new_txt_name = new_file_name + '.txt'

    # check if jpg or png file
    if file.endswith(".jpg"):
      new_img_name = new_file_name + '.jpg'
      shutil.move(original_file_name + '.jpg', new_img_name)
    else:
       shutil.move(original_file_name + '.png', new_img_name)
       



def main():

  # loop through all mesh files
  for file in os.listdir(MESH_DIRECTORY):
    # check if file is a mesh
    if file.endswith(".stl"):
      print("Processing file: " + file + "...")

      # file format is (classnumber)_filename.stl

      # get class number
      # class number may be 1 or 2 digits
      label = file.split('_')[0]

      # remove '(' and ')' from label
      label = label.replace('(', '')
      label = label.replace(')', '')

      # remove .stl from mesh name
      mesh_name = file.split('_')[1]
      mesh_name = mesh_name.replace('.stl', '')
      # check if there is a space in the mesh name
      if ' ' in mesh_name:
        # replace with underscore
        mesh_name = mesh_name.replace(' ', '_')
     
      # lower case
      mesh_name = mesh_name.lower()
      #add class number to mesh name
      mesh_name = label + '_' + mesh_name
     
      
      mesh_file = MESH_DIRECTORY + file # get mesh file
      
      mesh = Mesh.from_file(mesh_file) # load mesh
      
      take_picture(mesh, mesh_name) # take picture
      
      #create_yolo_label(label, mesh_name) # create yolo label
      
      augment_no_box(label, mesh_name) # augment
    
      add_background_to_img(mesh_name) # add background to image

    # split_data()
    split_data_img_class()

   

if __name__ == "__main__":
  main()

