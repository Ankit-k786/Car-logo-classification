"""Infernce for classify the logo
"""
from PIL import Image
from keras.models import load_model
import numpy as np
import time
from utils import constant

def predict_logo(image_path, model):
    """_summary_

    Args:
        image_path (String): Path of the logo which need to be calassify.
        model:Pretrained model
    """
    try:
        im = Image.open(image_path).convert("RGB")
        im = im.resize((50, 50))
        im = np.reshape(im, (1, 50, 50, 3))
        im = im.astype('float32') / 255
        start = time.time()
        pred = (model.predict(im))
        end = time.time()
        pred = list(pred[0])
        max_ind = pred.index(max(pred))
        print("==============================================")
        print('This logo belogs to class :- ', constant.classes[max_ind])
        print("The time of execution:", (end-start) * 10**3, "ms")
        print("===============================================")
    except Exception as e:
            print("Exception: {}".format(e))

if __name__ == "__main__":
     #Set the path
    image_path = './data/val/Audi.common/Audi_0ecdf0d4-7356-4b6c-a26b-365fa5c8b80a.jpg_2ac72fa5-f61a-4a9f-916a-9b6bfd680683.jpg'
    model_path = 'logo_classify_weight.h5'

    #Load the model
    model = load_model(model_path)
    predict_logo(image_path, model)

