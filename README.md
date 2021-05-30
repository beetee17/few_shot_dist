This repo contains the code used to train a few shot model on the Fine-Grained Visual Classification (FGVC) of Aircraft dataset. It also contains code of a simple web-service that can be used to demonstrate a model's prediction results with user uploaded images, as well as code to containerise the web-service into a Docker image.

A sample model may be found in the docker\app folder; the file is named *model.h5*. This model was created by fine-tuning the VGG16 pretrained model on the FGVC aircraft dataset.

# Environments
The codebase was developed with Python 3.9.
Install requirements as follows:  

```
pip install -r path\to\docker\requirements.txt
```

# Preprocessing
To train a few shot model, the following needs to be done: 
1. Ensure all images are of similar dimensions (and converted to 3 channel images if grayscale)
   
2. Four *.h5* files - *train_pairs.h5*, *val_pairs.h5* and *train_labels.h5*, *val_labels.h5* - are to be created
   - The *pairs.h5* files contain pairs of images as np arrays to be fed into our siamese network, while *labels.h5* files contain a single np array of 1s and 0s that correspond to the labels of the image pairs (1 if they are of the same class, and 0 otherwise)
  
3. Two *.h5* files - *X_test.h5* and *Y_test.h5* - are to be created for testing the model
   - *X_test* contains individual images (3 channel and similar dimensions to train/validation images) as np arrays
   - *Y_test* is a np array that contains the class names/labels corresponding to each image in *X_test*
  
4. Save the files with a similar h5 dataset name (i.e. *'train_pairs'* for *train_pairs.h5*, etc) for loading purposes

The *preprocess.py* utility script in the *Utils* folder may provide some help in padding/resizing images and pairing up the images.
   
# Training/Testing a Few Shot Model
Make sure to have done the required preprocessing as explained above.  

Then, in the command prompt, run the following:  

```
python path\to\main\main.py
```

*main.py* is an interactive script, so follow the onscreen instructions to either create a new model, train an existing model, or evaluate it on various one-shot tasks!

# Inference
In the command prompt, run the follownig:
```
python path\to\main\inference.py <path to images folder> <path to model.h5 file>
```
Do ensure that your images folder is formatted like so:

<pre>
folder  
   └── query  
       ├── foo.png   
       └── bar.jpg  
   └── support  
       ├── label1_foo.png  
       ├── label1_bar.jpg  
       ├── label2_foo.png  
       └── label2_bar.png 
       ... 
</pre>
See the example folder for more details.  
The program will pair the query images with each of the support images and output its similarity score.

# Running the Web-Service

In the command prompt, run the following:  

```
python path\to\docker\app\main.py
```

# Containerising the Web-Service
In the command prompt, run the following:  

```
cd path\to\docker\
docker build -t <image name> .
```

To run the web service image,  

```
docker run -p 8000:8000 <image name>
```

