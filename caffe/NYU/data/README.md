# README for NYU dataset preparation


## data basic structure
Basic structure should be like this:
-------------------------------------
./--
   |
   ----image folder
   |   |
   |   ----image files
   |
   ----label folder
   |   |
   |   ----label files
   |
   ----train.txt
   |
   ----test.txt

train.txt and test.txt are just filenames of images to be trained and tested(suffix not included).


Here are the steps of data preparation for both training and testing

## transform original .mat
During training process, we need png image files as input images,
.mat depth files as input labels.
Input images should be size of (304, 228, 3) and depth map should be
size of (160, 128).

Here, just run:

```
python createData.py
```

everything is done!
We'll get four directories:
train_image
train_label
test_image
test_label

warning:
```
data_path = 'nyu_depth_v2_labeled.mat'
split_path = 'splits.mat'
```
should be set where your actual data is.



## create train, test split
For now, I just use those 795 images for training without automatic testing. And
so, train and test split is just as official's arrangement.
Future we'll need data agumentation. We'll need our own train/test split then.
```
python createSplit.py data_path suffix
```
data_path is where your image data is,
suffix is where what your image suffix.



## get RMSE value
After training, we'll calculate error metrics.
For now, I only implemented RMSE(More metrics
will be added in the future).
usage:
```
python calcRMS.py train_score_path test_score_path
```
`train_score_path` and `test_score_path` are folders
after testing the model(which contains three kinds files:
image files, lable files, score files). Thay can be generated
by script in ../resnet/checkcaffeIO_batch.py.

