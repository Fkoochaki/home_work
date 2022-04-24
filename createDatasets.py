"""
In this script I read videos and create a train and validation dataset.
"""


import cv2
import json
import glob
import os
import shutil
import pickle as plk
import sys
import numpy as np
import random
import pdb


def readVid(vid_path, annot_path, data_names, data_labels, output_path="images"):
    """
    This script reads a video file and based on the annotations,
    I extract the frames which the obj_class is 'Needle"
    Then I set the labels to 1 if the 'orientation' is 'grabbed', otherwise to 0
    """

    # Extracting the video name
    vid_name = vid_path[-22:-4]

    # Creating a folder to write the frames
    frame_output_path = os.path.join(output_path, vid_name)

    try:
        shutil.rmtree(frame_output_path)
    except:
        pass

    os.makedirs(frame_output_path)
    
    # Loading the corresponding annotation
    with open(annot_path + vid_name + ".json") as f:
        annot = json.load(f)

    # Load the video
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()

    count = 0
    while success:
        if str(count) in annot: # if the frame number is available in annotation
            
            objc = annot[str(count)]["obj_class"]
            orn = annot[str(count)]["orientation"]
            
            if objc == "needle" and "grabbed" in orn:
                path = frame_output_path + "/frame" + str(count) + "_1.jpg"
                cv2.imwrite(path, image)
                data_names.append(path)
                data_labels.append(1)
            else:
                path = frame_output_path + "/frame" + str(count) + "_0.jpg"
                cv2.imwrite(path, image)
                data_names.append(path)
                data_labels.append(0)
                
        # Reading the next frame
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


if __name__ == "__main__":

    # Path to videos and annotations
    dir_path_vids = "Release_v1/videos/fps1/"
    dir_path_annots = "Release_v1/annotations/bounding_box_gt/"

    # Extracting frames from all videos
    all_data_names = []
    all_data_labels = []
    
    all_vids_path = glob.glob(dir_path_vids + "*.mp4")

    for vid_path in all_vids_path[:]:
        readVid(vid_path=vid_path, annot_path=dir_path_annots, data_names=all_data_names, data_labels=all_data_labels)

    print("--- All data ---")
    print("Len all data: ", len(all_data_names))
    print("Len all labels: ", len(all_data_labels))
    print("Len 0 lbl: ", all_data_labels.count(0))
    print("Len 1 lbl: ", all_data_labels.count(1))

    # Extracting some data for validation (balance between 0 and 1)
    # In order to have balance data I select a percent of label which is min
    val_percent = 0.1
    valNum = int(val_percent * min(all_data_labels.count(0), all_data_labels.count(1)))
    all_data_labels_np = np.asarray(all_data_labels)
    all_one_indices = np.where(all_data_labels_np == 1)[0]
    all_zero_indices = np.where(all_data_labels_np == 0)[0]
    rnd_index_ones = np.random.choice(all_one_indices, valNum, replace=False)
    rnd_index_zeros = np.random.choice(all_zero_indices, valNum, replace=False)

    val_names = []
    val_labels = []

    for ind in rnd_index_ones:
        val_names.append(all_data_names[ind])
        val_labels.append(1)
    
    for ind in rnd_index_zeros:
        val_names.append(all_data_names[ind])
        val_labels.append(0)

    print("\n--- Val data ---")
    print("Num of val data: ", valNum)
    print("Len val data: ", len(val_names))
    print("Len val labels: ", len(val_labels))
    print("Len 0 lbl: ", val_labels.count(0))
    print("Len 1 lbl: ", val_labels.count(1))

    # Removing the val data from all data
    all_data_names2 = []
    all_data_labels2 = []
    for ind, item in enumerate(all_data_names):
        if not ind in rnd_index_zeros and not ind in rnd_index_ones:
            all_data_names2.append(all_data_names[ind])
            all_data_labels2.append(all_data_labels[ind])

    print("\n--- All data after removing val data ---")
    print("Len all data: ", len(all_data_names2))
    print("Len all labels: ", len(all_data_labels2))

    # Creating balanced train data based on the label with min freq
    random.shuffle(all_data_names2)
    all_data_labels2_np = np.asarray(all_data_labels2)
    all_zero_indices = np.where(all_data_labels2_np == 0)[0]
    all_one_indices = np.where(all_data_labels2_np == 1)[0]
    sample_indices_zeros = np.random.choice(all_zero_indices, all_one_indices.shape[0], replace=False)

    train_names = []
    train_labels = []
    for ind in sample_indices_zeros:
        train_names.append(all_data_names2[ind])
        train_labels.append(all_data_labels2[ind])
    
    for ind in all_one_indices:
        train_names.append(all_data_names2[ind])
        train_labels.append(all_data_labels2[ind])

    print("\n--- Train data ---")
    print("Len train data: ", len(train_names))
    print("Len train labels: ", len(train_labels))
    print("Len 0 lbl: ", train_labels.count(0))
    print("Len 1 lbl: ", train_labels.count(1))
   
   # Dumping train/val data
    with open("train_data.plk", "wb") as fout:
        plk.dump(train_names, fout)

    with open("val_data.plk", "wb") as fout:
        plk.dump(val_names, fout)

    with open("train_lbl.plk", "wb") as fout:
        plk.dump(train_labels, fout)

    with open("val_lbl.plk", "wb") as fout:
        plk.dump(val_labels, fout)
