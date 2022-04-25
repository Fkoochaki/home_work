## home_work

### Data generation
For generating the training/validation data please run the createDatasets.py script.
```python python createDatasets.py
It creates a folder which the name is "images" and inside of it there is one folder belong to each video. In the folders corresponding to videos, I save the frames which will be used for training.
In addition, this sceript generate train/validation data list and their corresponding labels. They are saved as pickle file. They are 'train_data.plk', 'train_lbl.plk', 'val_data.plk' and 'val_lbl.plk'.
Please set the path to videos and annotations inside the script.

Inside this script I read all videos and if the obj_calss in annoatation is 'needle' I check the 'orientation'. If the 'orientation' is grabbed I set the label as 1, otherise I set it as 0. Also, the labels is the last character of saved images in images folder.
Once, I was generating the data, I noticed we have inbalancing between classes. Specifically we have more data for negative calss(drop). Since I have a short time for this project, I just balanced them inside this script.

### Dataset
This script is the dataset script for loading the data during the training.

### Main
This script is for training and inference.
For training please pass 'train' as command line input. 'python main train'
For inefernce please pass 'infer' as command line input. 'python main infer'

For training, I tried an image classification model based Vision transfomer. I am loading a pretrained model and I change the last layer (classification layer) to a linear layer for two neurouns since the problem is binary classification. Then I am fine tunning the vision transformer with the new added layer for classification. The performance is not satisfactory, I would try an objecet detection model I didn't have time to do it. Therefore, I wrap up he project with this model only. After training, it will save the model inside the same folder.

For inference, I am loading the fine-tuned model and then I read each frame of the given video. Each frame is a opencv format which I conver them to pillow image and after preprocessign, each frame is fed to the model and I print the predicted label. Please set the name of the video for ineference inside the script. There is an issue with the given test data and opencv cannot read both of them, in the case it works weill with train data. I tested the inefernce code with training data.

### Notes
I felt in most of video we have a constant background, I tried to remove them from videos with opencv, but teh result was not right for all videos. In order to save time, I didn't dig into it.
