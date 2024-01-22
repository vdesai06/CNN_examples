import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Dropout, Input
from keras.metrics import MeanSquaredError, RootMeanSquaredError
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras_tuner as kt
from keras_tuner import HyperParameters
import os


hp = HyperParameters()

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


# Image and dark matter value filepaths
imgs_filepath = r"/.../RefL0100N1504_Subhalo"
DM_path = r"/.../Matched_Eagle_COSMA.txt"
new_folder = r"/.../Galaxy_imgs"
rot_folder = r"/.../rotated_images"

dm = pd.read_csv(DM_path, sep=", ", header=None, names=["logDM", "ID"])
dm.logDM = np.log10(dm.logDM)

imgs = Path(new_folder)
filepath_imgs = pd.Series(list(imgs.glob("*.png")), name="file paths").astype(str)
filepath_imgs = filepath_imgs.to_frame()
filepath_imgs['ID'] = filepath_imgs["file paths"].apply(lambda x: x.split('_')[-1][:-4])
filepath_imgs['ID'] = filepath_imgs['ID'].astype(int)
images = filepath_imgs.merge(dm, left_on='ID', right_on='ID')
images = images.drop("ID", axis=1)

rot_imgs = Path(rot_folder)
rot_filepath_imgs = pd.Series(list(rot_imgs.glob("*.png")), name="file paths").astype(str)
rot_filepath_imgs = rot_filepath_imgs.to_frame()
rot_filepath_imgs["ID"] = rot_filepath_imgs["file paths"].apply(lambda x: x.split("_")[-3])
rot_filepath_imgs["ID"] = rot_filepath_imgs["ID"].astype(int)
rot_images = rot_filepath_imgs.merge(dm, left_on="ID", right_on="ID")
rot_images = rot_images.drop("ID", axis=1)

all_images = pd.concat([images, rot_images], ignore_index=True)
all_images = all_images.sample(frac=1.0, random_state=42)

train_df, test_df = train_test_split(all_images, train_size=0.8, random_state=42)
t = len(train_df)
tt = len(test_df)
print(f"The length of the training data is {t} and the length of the test data is {tt}")

gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.25)
gen_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# generate the arrays that flow into the network using flow_from_dataframe
train_imgs = gen_train.flow_from_dataframe(dataframe=train_df,
                                           x_col="file paths",
                                           y_col="logDM",
                                           target_size=(256, 256),
                                           color_mode="rgb",
                                           class_mode="raw",
                                           batch_size=64,
                                           shuffle=True,
                                           seed=42,
                                           subset="training"
                                           )

val_imgs = gen_train.flow_from_dataframe(dataframe=train_df,
                                         x_col="file paths",
                                         y_col="logDM",
                                         target_size=(256, 256),
                                         color_mode="rgb",
                                         class_mode="raw",
                                         batch_size=64,
                                         shuffle=True,
                                         seed=42,
                                         subset="validation"
                                         )

test_imgs = gen_test.flow_from_dataframe(dataframe=test_df,
                                         x_col="file paths",
                                         y_col="logDM",
                                         target_size=(256, 256),
                                         color_mode="rgb",
                                         class_mode="raw",
                                         batch_size=32,
                                         shuffle=False
                                         )


# Build the model framework, without setting the hyperparameters - perform Bayesian Optimisation
def create_model(hp):
    DM_model = Sequential()

    DM_model.add(Conv2D(hp.Choice('filters_1', values=[32, 64, 128]), (3, 3),
                        padding=hp.Choice('padding_1', values=['valid', 'same']),
                        activation="relu", kernel_initializer="he_uniform"))
    DM_model.add(MaxPool2D((2, 2)))
    DM_model.add(Dropout(hp.Choice("dropout_1", values=[0.0, 0.1, 0.2, 0.3])))

    DM_model.add(Conv2D(hp.Choice('filters_2', values=[32, 64, 128, 256]), (3, 3),
                        padding=hp.Choice('padding_2', values=['valid', 'same']),
                        activation="relu", kernel_initializer="he_uniform"))
    DM_model.add(MaxPool2D((2, 2)))
    DM_model.add(Dropout(hp.Choice("dropout_2", values=[0.1, 0.2, 0.3, 0.4])))

    DM_model.add(Conv2D(hp.Choice('filters_3', values=[32, 64, 128, 256]), (3, 3),
                        padding=hp.Choice('padding_3', values=['valid', 'same']),
                        activation="relu", kernel_initializer="he_uniform"))
    DM_model.add(MaxPool2D((2, 2)))
    DM_model.add(Dropout(hp.Choice("dropout_3", values=[0.3, 0.4, 0.5])))

    DM_model.add(GlobalAveragePooling2D())
    hp_neurons = hp.Choice("neurons", values=[32, 64, 128, 256])
    DM_model.add(Dense(hp_neurons, activation="relu", kernel_initializer="he_uniform"))
    DM_model.add(Dense(hp_neurons, activation="relu", kernel_initializer="he_uniform"))

    DM_model.add(Dense(1, activation="linear"))
    lr = hp.Choice("learning_rate", values=[0.01, 0.001])
    optimizer = Adam(learning_rate=lr)
    DM_model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
    return DM_model


dir_path = r"/.../data"
CNN_optimiser = kt.tuners.BayesianOptimization(create_model, objective="mse", max_trials=150, directory=dir_path,
                                               project_name="CNN_optimising")

es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

CNN_optimiser.search(train_imgs, epochs=150, batch_size=64, validation_data=val_imgs, callbacks=[es])


best_hp = CNN_optimiser.get_best_hyperparameters(num_trials=1)[0]
print("Best filters_1 number is: {}".format(best_hp.get("filters_1")))
print("Best dropout_1 number is: {}".format(best_hp.get("dropout_1")))
print("Best filters_2 number is: {}".format(best_hp.get("filters_2")))
print("Best dropout_2 number is: {}".format(best_hp.get("dropout_2")))
print("Best filters_3 number is: {}".format(best_hp.get("filters_3")))
print("Best dropout_3 number is: {}".format(best_hp.get("dropout_3")))
print("Best neurons number is: {}".format(best_hp.get("neurons")))
print("Best learning rate is: {}".format(best_hp.get("learning_rate")))


