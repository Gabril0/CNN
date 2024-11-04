import os
import shutil

#this code makes all the training/test data into one folder, with masks into another folder to make the process easier
def arrange_data(target_path, path_list):
    img_path = os.path.join(target_path, "images")
    mask_path = os.path.join(target_path, "masks")
    if os.path.exists(target_path):
        print("Target path already exists. Please delete it first if you want to re-arrange the data, exiting...")
    else:
        print("Path does not exist.\nArranging data...")
        os.makedirs(img_path)
        os.makedirs(mask_path)
        for folder in path_list:
            append_paths(folder, img_path, mask_path)
            
def append_paths(folder, img_path, mask_path):
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if file.endswith(".png"):
                if "02-mask" in folder.lower():
                    shutil.copy(os.path.join(folder, file), mask_path)
                else:
                    shutil.copy(os.path.join(folder, file), img_path)
            else:
                print("Skipping non-PNG file:", os.path.join(folder, file))
    else:
        print("Error: {} could not be found".format(folder))

def __init__(self):
        pass