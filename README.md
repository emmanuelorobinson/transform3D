# 3D->2D Dataset Generation Pipleine

This repository contains the code for generating 3D->2D datasets for the training of YOLO models.

## Requirements
- Python 3.8.10

## Installation
pip install -r requirements.txt

## Usage
### 1. Mesh Folder
Create a folder named "mesh" in the root directory of the project.
The mesh folder should contain the .stl files of the 3D models. The .obj files should be named as the following format:
```(class_number)_filename.stl```
where class_number is the class number of the object. For example, if the class number of the object is 1, the file name should be ```(1)_filename.stl```.

### 2. Dataset Folder
Create a folder named "dataset" in the root directory of the project.
This folder would contain the generated dataset. This includes a png file for each object and a .txt file for each object containing the bounding box coordinates.

### 3. Background Images
Create a folder named "background" in the root directory of the project.

### 4. Run the code
```python transform.py```

### 5. Playground
The playground folder contains the code for training and testing on the generated dataset. This are just for testing purposes and are not part of the dataset generation pipeline.