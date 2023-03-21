from keras.utils import to_categorical
import numpy as np
from utils import helper, constant
from keras.models import load_model
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def test(train_folder_path,model):
    """_summary_

    Args:
        train_folder_path (_Str): path of testing data folder.

    """
    try:
        images, labels = helper.get_image_and_labels(test_folder_path)
        print(np.shape(images))
        print(np.shape(labels))
        X_test = helper.reshape_normlize(images)
        Y_test = to_categorical(labels, num_classes=len(constant.classes))
        Y_pred = (model.predict(X_test))
        true=[]
        pred=[]
        for i in range(len(Y_test)):
            true.append(list(Y_test[i]).index(max(list(Y_test[i]))))
            pred.append(list(Y_pred[i]).index(max(list(Y_pred[i]))))
        labels = constant.classes
        l=np.arange(28)
        helper.save_confusion_matrix(true, pred)
        accuracy=accuracy_score(true,pred).round(2)
        p,r,f,u=precision_recall_fscore_support(true, pred, average=None,labels=l)
        print('=====================================================================================')
        print('Test Accuracy = ',accuracy*100,"%")
        print('=====================================================================================')
        for i in range(len(constant.classes)):
            print('Class: ',constant.classes[i],' Num_sample: ',u[i]," Precision:",p[i].round(2),' Recall:',r[i].round(2)," Fscore:",f[i].round(2))
            print('==================================================================================')
    except Exception as e:
            print("Exception: {}".format(e))
        


    
    


if __name__ == "__main__":
    test_folder_path = './data/val/'
    model_path = 'logo_classify_weight.h5'
    model = load_model(model_path)
    test(test_folder_path,model)
