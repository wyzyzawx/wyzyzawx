import numpy as np
import tensorflow as tf
import scipy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
import os
import csv
from EEGModels import EEGNet
import matplotlib.pyplot as plt
import datetime
matplotlib.use('Agg')

def shift_labels(labels):
    nb_classes = np.unique(labels)
    nb_classes = len(nb_classes)-1
    shifted_labels = np.array([int(np.ceil((nb_classes*((label-np.min(labels))))/(np.max(labels)-np.min(labels))))for label in labels])
    return shifted_labels

def get_data(subject):
    data_mat = scipy.io.loadmat('Official_Preprocessed_Data/S' + str(subject) + '_data.mat')
    labels_mat = scipy.io.loadmat('Official_Preprocessed_Data/S'+ str(subject)+'_labels.mat')
    for key, value in data_mat.items():
        if key == 'data':
            data = value
    for key, value in labels_mat.items():
        if key == 'labels':
            labels_current = value
    labels_current = [int(x) for x in labels_current[0]]
    data = data.squeeze()
    matrix_current = np.empty((0, 33, 1500))
    j = []
    for i in range(0, len(data)):
        data[i] = np.swapaxes(data[i], 0, 2)
        data[i] = np.swapaxes(data[i], 1, 2)
        j.append(len(data[i]))
        matrix_current = np.concatenate((matrix_current, data[i][:, :, 500:2000]), axis=0)
    labels_aux = []
    for i in range(len(labels_current)):
        for iteratie in range(j[i]):
            labels_aux.append(labels_current[i])
    labels_aux = shift_labels(labels_aux)
    return matrix_current, labels_aux



def k_fold_train_subject_model(subject, data, labels, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    # Create a directory for the subject if it doesn't exist
    fig_subject_dir = f'figures/likeness_degree_individual_users_{k}_folds/Subject_{subject}'
    models_subject_dir = f'paper_models/likeness_degree_individual_users_{k}_folds/Subject_{subject}'
    os.makedirs(fig_subject_dir, exist_ok=True)
    os.makedirs(models_subject_dir, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
        print(f"Training fold {fold+1}/{k} for subject {subject}")
        log_dir = f"./paper_logs/identity_classifier/model_fold_{fold+1}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        x_train, x_test = data[train_idx], data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        y_train, y_test = to_categorical(y_train), to_categorical(y_test)
        nb_classes = y_train.shape[1]

        model = EEGNet(nb_classes=nb_classes, Chans=33, Samples=1500, dropoutRate=0.5, kernLength=256, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout')
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

        
        def get_tensorboard_callback(log_dir):
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
            return tensorboard_callback

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=50, verbose=0, mode="auto", restore_best_weights=True),
            ModelCheckpoint(f'{models_subject_dir}/S{subject}_fold_{fold+1}.h5', save_best_only=True, verbose=1),
            get_tensorboard_callback(log_dir)
        ]

        history = model.fit(x_train, y_train, batch_size=32, epochs=200, verbose=2, callbacks=callbacks, validation_data=(x_test, y_test))
        fold_accuracies.append(max(history.history['val_accuracy']))

        # Plot training & validation accuracy and loss
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title(f'Subject {subject} - Fold {fold+1} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title(f'Subject {subject} - Fold {fold+1} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{fig_subject_dir}/Fold_{fold+1}_Plot.png")
        plt.close()

    return fold_accuracies


def save_results_to_csv(subject_results, k):
    filepath=f'paper_csvs/likeness_degree_individual_users/{k}_fold_results.csv'
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Subject', 'Mean Accuracy', 'Std Accuracy'])
        for subject, accuracies in subject_results.items():
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            writer.writerow([subject, mean_accuracy, std_accuracy])
            print(f"Subject {subject}: Mean Accuracy = {mean_accuracy:.4f}, Std = {std_accuracy:.4f}")
            

def main():
    subjects = range(1, 25) 
    k = 2
    subject_results = {}

    for subject in subjects:
        data, labels = get_data(subject)
        fold_accuracies = k_fold_train_subject_model(subject, data, labels, k=k)
        subject_results[subject] = fold_accuracies

    save_results_to_csv(subject_results, k)

if __name__ == "__main__":
    main()
