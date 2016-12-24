# normal, log_path, mask

I have trained model with three strategies:
1. original data
2. depth with log transforamtion
3. mask depth larger than 70m.


## A little story 
I have been confused with "mask out depth larger than 70m".
At first, I handled by convert depth larger than 70m to 70m.
This causes the model can not converge. Later, I tracked older
depth prediction paper and found that they mask out by just
not calculating those prediction errors!!!
Since it is outdoor scene, images are highly possible with sky
which is infinite far away(depth value around 80) and could affect
model training. So, I trained the model with log scale transformation.
However, it seems that I did not get any good result. Finally, I trained
the model by masking out those sky areas(you can see this in prototxt file).
Result is still to be evaluated...


## Need help
I have been troubled with data agumenation. I doubt that my codes are somewhere
wrong. How can I do it properly?
1. image is easy to transform, but depth map is not the same type as image including
size and value. When doing translation, I guess we have to resize them to the same size.
However, depth map values are in rationals. I did not find any proper functions to 
resize(and rotate) rational matrix.
When doing scale, depth should be divided by the scale ratio? Is this OK?


## Pre-request
This is a FCN-segmentation style caffe. So we rely on python layer support. In the mean
time, we need python path link to current path(to find related python file).


##Usage
This is quite simple, just go to path(normal/log_depth/mask), run:
```
python solve.py gpu-id [2>&1 | tee log.txt]
```


##Notifications
Training and testing can be combines in one file by parsing parameters in prototxt, but I'm
using an older version caffe which seems to fail to parse parameters(I do not know why yet).
