import os
import numpy as np
from PIL import Image
from utils import constant
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def get_image_and_labels(path):
    """_summary_

    Args:
        path (Str): path of train folder

    Returns:
        list: list of images and labels
    """
    try:
        images = []
        labels = []
        train_folders = os.listdir(path)
        for label in train_folders:
            image_folder_path=path+label+"/"
            img_list=os.listdir(image_folder_path)
            for img_name in img_list:
                im = Image.open(image_folder_path + img_name).convert("RGB")
                im = im.resize((constant.Img_Size,constant.Img_Size))
                im = np.array(im).flatten() 
                images.append(np.array(im , dtype='uint8' ))
                labels.append(np.array(constant.classes.index(label)))
        return images,labels
    except Exception as e:
            print("Exception: {}".format(e))

def suffel_data(images, labels):
    """_summary_

    Args:
        images (list): train_images
        labels (list): train_labels

    Returns:
        _type_: images and labels
    """
    try:
        imgs, label = shuffle(images, labels, random_state=42)
        data = [imgs, label]
        return data
    except Exception as e:
            print("Exception: {}".format(e))

def reshape_normlize(X):
    try:
        X = np.reshape(X,(np.shape(X)[0],constant.Img_Size,constant.Img_Size,3))
        X = X.astype('float32') / 255
        return X
    except Exception as e:
            print("Exception: {}".format(e))

def save_confusion_matrix(true, pred):
    try:
        conf = confusion_matrix(true, pred)
        target_names = constant.classes
        disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=target_names)
        plt.rcParams["figure.figsize"] = (20,15)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=85,)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", pad_inches=500)
    except Exception as e:
            print("Exception: {}".format(e))

    
