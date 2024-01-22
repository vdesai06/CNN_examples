import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import callbacks
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats
import os
import time

start = time.time()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
all_images = all_images.sample(frac=1.0, random_state=25)
print(all_images)

train_df, test_df = train_test_split(all_images, train_size=0.8, random_state=25)
t = len(train_df)
tt = len(test_df)
print(f"The length of the training data is {t} and the length of the test data is {tt}")


gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.25)
gen_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


# generate the arrays that flow into the network using flow_from_dataframe
train_imgs = gen_train.flow_from_dataframe(dataframe=train_df,
                                           x_col="file paths",
                                           y_col="logDM",
                                           target_size=(256, 256),
                                           color_mode="rgb",
                                           class_mode="raw",
                                           batch_size=64,
                                           shuffle=True,
                                           seed=25,
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
                                         seed=25,
                                         subset="validation"
                                         )

test_imgs = gen_test.flow_from_dataframe(dataframe=test_df,
                                         x_col="file paths",
                                         y_col="logDM",
                                         target_size=(256, 256),
                                         color_mode="rgb",
                                         class_mode="raw",
                                         batch_size=64,
                                         shuffle=False
                                        )


# Build the network
inputs = tf.keras.Input(shape=(256, 256, 3))
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="valid")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="valid")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="linear")(x)

epochs = 650
patience = 90
learning_rate = 0.001

DM_model = tf.keras.Model(inputs=inputs, outputs=outputs)
DM_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse",
                 metrics=[tf.keras.metrics.MeanSquaredError()])

data_path = r"/.../best_model_rgb.h5"

early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(data_path, monitor='val_loss', save_best_only=True, mode='min')

DM_fitted_model = DM_model.fit(train_imgs, validation_data=val_imgs, epochs=epochs, callbacks=[early_stop, checkpoint])


predicted_DM = np.squeeze(DM_model.predict(test_imgs))
sorted_pred = sorted(predicted_DM)
sorted_pred = sorted_pred[2:]
preds = np.array([x for x in predicted_DM if x in sorted_pred])
np.save(r"/.../cnn_preds.npy", preds)

actual_DM = test_imgs.labels
sorted_act = sorted(actual_DM)
sorted_act = sorted_act[2:]
acts = np.array([x for x in actual_DM if x in sorted_act])
np.save(r"/.../cnn_acts.npy", acts)


rmse = np.sqrt(DM_model.evaluate(test_imgs, verbose=1))
print("Test RMSE: {:.5f}".format(rmse[0]))
r2 = r2_score(acts, preds)
print("Test R^2 score: {:.5f}".format(r2))

# plot the validation and training loss for each epoch
loss = DM_fitted_model.history["loss"]
np.save(r"/.../cnn_loss.npy", loss)
val_loss = DM_fitted_model.history["val_loss"]
np.save(r"/.../cnn_val_loss.npy", val_loss)

epochs = range(len(loss)-2)
plt.figure()
plt.rcParams.update({'font.size': 12})
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.plot(epochs, loss[2:], "b", label="Loss (MSE)")
plt.plot(epochs, val_loss[2:], "r", label="Validation loss (MSE)")
plt.legend()
plt.savefig("loss_plot_report.png")

# best weights and corresponding mse
DM_model.load_weights(data_path)

best_predictions = np.squeeze(DM_model.predict(test_imgs))
srtd_bp = sorted(best_predictions)
srtd_bp = srtd_bp[2:]
bests = np.array([x for x in best_predictions if x in srtd_bp])
print(f"The best predictions are: {bests}")
np.save(r"/.../cnn_bests.npy", bests)

true_vals = test_imgs.labels
srtd_true = sorted(true_vals)
srtd_true = srtd_true[2:]
trues = np.array([x for x in true_vals if x in srtd_true])
print(f"The true values are: {trues}")
np.save(r"/.../cnn_trues.npy", trues)

best_rmse = np.sqrt(DM_model.evaluate(test_imgs, verbose=1))
print("Best Test RMSE: {:.5f}".format(best_rmse[0]))
r2_best = r2_score(bests, trues)
print("Best test R^2 score: {:.5f}".format(r2))

end = time.time()
total = end - start
print("\n" + "Total time taken: " + str(total) + " seconds")
