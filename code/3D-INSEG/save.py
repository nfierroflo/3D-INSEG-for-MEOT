import os
import shutil

def redistribute_images(left_imgs_path,right_imgs_path,destination_folder):
    #file names of a folder sorted
    left_imgs = sorted(os.listdir(left_imgs_path))
    right_imgs = sorted(os.listdir(right_imgs_path))
    #recover the file names
    for (imfile1, imfile2) in zip(left_imgs, right_imgs):
        # Create the destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)
        subdir=os.path.join(destination_folder,imfile1.split('_')[2])
        os.makedirs(subdir, exist_ok=True)
        #move the files to another folder
        shutil.copy(os.path.join(left_imgs_path,imfile1),os.path.join(subdir,imfile1))
        shutil.copy(os.path.join(right_imgs_path,imfile2), os.path.join(subdir,imfile2))

        shutil.copy(os.path.join(left_imgs_path,imfile1),os.path.join(subdir,"im0.jpg"))
        shutil.copy(os.path.join(right_imgs_path,imfile2), os.path.join(subdir,"im1.jpg"))
    
