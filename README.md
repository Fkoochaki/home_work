## home_work

### Data generation
Please run the createDatasets.py, it creates a folder which name is "images" and inside of it there is one folder belong to each video. In the folders corresponding to videos, I save the frames which will be used for training.
In addition, this sceript generate train/validation data list and their corresponding labels. They are saved a pickle file. They are 'train_data.plk', 'train_lbl.plk', 'val_data.plk' and 'val_lbl.plk'.
Please set the path to videos and annotations inside the script.

### Dataset
This script is the dataset script for loading the data during the training.

### Main
This script is for training and inference.
For training please pass 'train' as command line input. 'python main train'
For inefernce please pass 'infer' as command line input. 'python main infer'
