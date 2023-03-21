from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import numpy as np
from utils import helper, constant, model_cfg
from keras.utils import to_categorical


def train(train_folder_path, n_epochs, batch):
    """_summary_

    Args:
        train_folder_path (_Str): path of training data folder
    """
    images, labels = helper.get_image_and_labels(train_folder_path)
    print(np.shape(images))
    print(np.shape(labels))
    data = helper.suffel_data(images, labels)
    X_train, X_val, y_train, y_val = train_test_split(
        data[0], data[1], test_size=0.1)
    X_train = helper.reshape_normlize(X_train)
    X_val = helper.reshape_normlize(X_val)
    Y_train =to_categorical(y_train, num_classes=len(constant.classes))
    Y_val = to_categorical(
        y_val, num_classes=len(constant.classes))
    model = model_cfg.make_model(constant.Img_Size, constant.Num_Class)
    # from keras.models import load_model
    # model_path = 'logo_classify_weight.h5'
    # model = load_model(model_path)
    print(model.summary())
    take_best_model = ModelCheckpoint(
        './model/logo_classify_weight.h5', verbose=0, save_freq=2, save_best_only=False)
    optimizer = Adam(learning_rate=0.0004)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['categorical_accuracy'])
    model.fit(X_train, Y_train, batch_size=batch,
              steps_per_epoch=X_train.shape[0]//batch, epochs=n_epochs, callbacks=[take_best_model], validation_data=(X_val, Y_val))


if __name__ == "__main__":
    train_folder_path = './data/train/'
    n_epochs = 25
    batch = 512
    train(train_folder_path, n_epochs, batch)
