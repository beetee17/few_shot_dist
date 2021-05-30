from PIL import Image
from tensorflow.keras.models import load_model
import os
import sys
import numpy as np
from Utils import preprocess

def get_prediction(path, model):

    support_set = list()
    queries = list()
    all_pairs = list()

    # for each file: read and write into the static/images dir
    # filter the files into support and query images by looking at the filenames
    # pad and resize the images to allow input into our model (280x200)px
    # feed the images into the model and return the predictions as a list

    files = [os.path.join(os.path.join(path, 'query'), file) for file in os.listdir(os.path.join(path, 'query'))]
    files.extend([os.path.join(os.path.join(path, 'support'), file) for file in os.listdir(os.path.join(path, 'support'))])

    query_files = os.listdir(os.path.join(path, 'query'))
    support_files = os.listdir(os.path.join(path, 'support'))

    for file in files:
     
        image_bytes = Image.open(file)
        
        image = np.array(image_bytes)
        image = image[:image.shape[0]-20, :]
        image_bytes = Image.fromarray(image)

        image = preprocess.pad_and_resize(image_bytes, desired_ratio=1.4, width=280, height=200)

        if 'support' in file:

            support_set.append(image)

        elif 'query' in file:
            queries.append(image)

    for query_img in queries:
        
        all_pairs.append([[query_img, support_img] for support_img in support_set])
    
    all_pairs = np.array(all_pairs)

    predictions = list()
 
    for query_pairs in all_pairs:

        prediction = model.predict([query_pairs[:,0], query_pairs[:,1]]).flatten()

        predictions.extend(list(map(float, prediction)))


    i=0
    for query in query_files:
        print('\n')
        for support in support_files:
            print(query, support)
            print(round(predictions[i], 4))
            i += 1

    return predictions

if __name__ == '__main__':
    
    args = sys.argv

    model = load_model(args[2])

    get_prediction(args[1], model)

