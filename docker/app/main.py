import io
import os
import shutil
import uvicorn
import requests
import numpy as np
import preprocess

from PIL import Image
from pathlib import Path
from typing import Optional, List
from tensorflow.keras.models import load_model


from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, File, UploadFile

### pip install fastapi
### pip install uvicorn[standard]

# cd C:\Users\Admin\few_shot_learning\aircraft\docker\app
# uvicorn main:app --reload-dir 

print('loading model...')
db = []
app = FastAPI()

path = os.path.dirname(os.path.realpath(__file__))
model = load_model(os.path.join(path, 'model.h5'))

app.mount("/static", StaticFiles(directory=os.path.join(path,"static")), name="static")

templates = Jinja2Templates(directory=os.path.join(path,'templates')


@app.post("/get_predictions/")
async def create_upload_files(request : Request, files: List[UploadFile] = File(...)):
    """Receives a list of files uploaded by the user via the form in root webpage. Saves the files into the static folder and also gets the model's predictions for each support/query pair. Returns a HTML Response with the prediction outputs and the filenames for visualisation purposes"""

    support_set = list()
    queries = list()
    all_pairs = list()
    filenames = list()

    # for each file: read and write into the static/images dir
    # filter the files into support and query images by looking at the filenames
    # pad and resize the images to allow input into our model (280x200)px
    # feed the images into the model and return the predictions as a list
    for file in files:
     
        contents = await file.read()

        file_name = os.getcwd() + '\\static\\images\\' + file.filename.replace('/', '\\')
        filenames.append(file_name.replace(os.getcwd() + '\\static\\', ''))
        Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)

 
        image_bytes = Image.open(io.BytesIO(contents))
        
        image = np.array(image_bytes)
        image = image[:image.shape[0]-20, :]
        image_bytes = Image.fromarray(image)

        image = preprocess.pad_and_resize(image_bytes, desired_ratio=1.4, width=280, height=200)

        image_bytes = Image.fromarray(image)
        image_bytes.save(file_name)

        if 'support' in file.filename:

            support_set.append(image)

        elif 'query' in file.filename:
            queries.append(image)

    for query_img in queries:
        
        all_pairs.append([[query_img, support_img] for support_img in support_set])
    
    all_pairs = np.array(all_pairs)

    predictions = list()
 
    for query_pairs in all_pairs:

        prediction = model.predict([query_pairs[:,0], query_pairs[:,1]]).flatten()
        predictions.append(list(map(float, prediction)))

    
    return  templates.TemplateResponse('predictions.html', {'request' :  request, 'filenames' : filenames, 'predictions' : predictions})


@app.get("/")
def read_root(request : Request):

    # ensure the static/images dir is cleared before user uploads any files 
    
    try:

        path = 'C:\\Users\\Admin\\few_shot_learning\\aircraft\\docker\\app\\static\\images'

        for dir_ in os.listdir(path):

            shutil.rmtree(path + '\\' + dir_)

    except Exception as e:

        print(e)

    return templates.TemplateResponse('home.html', {'request' : request})
    

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='0.0.0.0')