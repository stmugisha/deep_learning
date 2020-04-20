# Custom helper functions file.
# Author: Stephen Mugisha

import os
import shutil

def create_dir(names: [str], path: str):
    """
    Creates the directories passed in the names arg list
    in the specified path within the current working directory.
    Args:
        `names`: A list of directory names to be created.
        `path`: The path where to create the directories.
    """
    for dir_name in names:
        new_dir = os.path.join(path, dir_name)
        img_path = os.path.join('./', new_dir)
        if not os.path.exists(img_path):
            os.mkdir(img_path)
        else:
            print(f'{img_path}: Exists!') 


def move_images(image_list: [str], source_dir: str, dest_dir: str):
    """
    Move images in image_list from source path to dest path.
    Args:
        `image_list`: A list of image files to be moved between directories.
        `source_dir`: The source/current directory holding the files.
        `dest_dir`: The new directory where the files will be transfered.
    """
    for img in image_list:
        shutil.move(source_dir+img, dest_dir)
        
