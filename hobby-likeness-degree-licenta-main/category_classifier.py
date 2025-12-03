import numpy as np
import os
import tensorflow as tf
import random
import csv
import scipy
import itertools
import datetime
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from EEGModels import EEGNet
from sklearn.metrics import confusion_matrix
from tensorflow.keras.mixed_precision import set_global_policy
import matplotlib
matplotlib.use('Agg')

set_global_policy('mixed_float16')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def shift_labels(labels):
    nb_classes = np.unique(labels)
    nb_classes = len(nb_classes)-1
    shifted_labels = np.array([int(np.ceil((nb_classes*((label-np.min(labels))))/(np.max(labels)-np.min(labels))))for label in labels])
    return shifted_labels


def inspect_batches(ds, num_batches=1000):
    for batch_index, (x, y) in enumerate(ds.take(num_batches)):
        labels_indices = tf.argmax(y, axis=1).numpy()
        unique, counts = np.unique(labels_indices, return_counts=True)
        print(f"Batch {batch_index + 1}: Unique labels - {unique}, Counts - {counts}")

def set_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)


from sklearn.model_selection import StratifiedShuffleSplit

def generate_stratified_splits_for_subjects(num_subjects, n_splits=5, test_size=0.2):
    all_subject_splits = {}
    for subject_id in range(1, num_subjects + 1):
        label_path = f'./database_npy/S{subject_id}_category_labels.npy'
        labels = np.load(label_path)
        labels = shift_labels(labels)  # Adjust labels as necessary
        
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=None)
        splits = list(sss.split(np.zeros(len(labels)), labels))
        all_subject_splits[subject_id] = splits
    
    return all_subject_splits



def load_data_on_demand(subject_id, num_classes, stratified_indices):
    """
    A generator function that loads data for a specific subject on demand, 
    yielding data for either training or validation based on the stratified indices.
    """

    data_path = f'./database_npy/S{subject_id}_data.npy'
    label_path = f'./database_npy/S{subject_id}_category_labels.npy'
    
    # Load labels to adjust and one-hot encode them
    labels = np.load(label_path)
    labels = shift_labels(labels)  # Ensure this adjusts labels correctly
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    for i in stratified_indices:
        data = np.load(data_path, mmap_mode='r')[i]  # Load specific data point
        yield tf.expand_dims(data, axis=-1), labels[i]




def create_dynamic_dataset(subject_ids, num_classes, fold_idx, stratified_indices_dict, is_train):
    datasets = []
    for subject_id in subject_ids:
        train_indices, test_indices = stratified_indices_dict[subject_id][fold_idx]
        indices = train_indices if is_train else test_indices
        # print(f"Subject {subject_id}, Indices Type: {type(indices)}, Length: {len(indices)}")  # Debug print

        gen_func = lambda subject_id=subject_id, is_train=is_train, num_classes=num_classes, stratified_indices=indices: load_data_on_demand(subject_id, num_classes, stratified_indices) 
        
        ds = tf.data.Dataset.from_generator(
            gen_func,
            output_types=(tf.float32, tf.float32),
            output_shapes=((None, None, 1), (num_classes,))
        )
        datasets.append(ds)
    
    combined_ds = datasets[0]
    for ds in datasets[1:]:
        combined_ds = combined_ds.concatenate(ds)
    
    return combined_ds.shuffle(10000).batch(256).prefetch(tf.data.AUTOTUNE)




def train_and_evaluate(train_ds, val_ds, fold_idx, figures_dir, model_save_path, num_classes=32, overwrite=False):
    if os.path.exists(model_save_path) and overwrite==False:
        print(f"Model already exists at {model_save_path}. Loading model...")
        model = tf.keras.models.load_model(model_save_path)
        history = None
    else:
        log_dir = f"./paper_logs/category_classifier/model_fold_{fold_idx+1}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if overwrite==True:
            print("Overwriting existing model...")
        else:
            print("Training new model...")
        model = EEGNet(nb_classes=num_classes, Chans=33, Samples=1500, dropoutRate=0.5, kernLength=256, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout')
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        def get_tensorboard_callback(log_dir):
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
            return tensorboard_callback
    
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True),
            ModelCheckpoint(filepath=model_save_path, save_best_only=True, verbose=1),
            get_tensorboard_callback(log_dir)
        ]

        history = model.fit(train_ds, epochs=500, validation_data=val_ds, callbacks=callbacks, verbose=1)

    # Directly evaluate the model on the validation dataset
    val_loss, val_accuracy_direct = model.evaluate(val_ds, verbose=0)
    print(f'Direct validation accuracy (post-training): {val_accuracy_direct*100:.2f}%')

    # Generate predictions and true labels for confusion matrix
    y_pred = []
    y_true = []

    # iterate over the dataset
    for eeg_batch, label_batch in val_ds: 
        # append true labels
        y_true.append(label_batch)
        # compute predictions
        preds = model.predict(eeg_batch)
        # append predicted labels
        y_pred.append(np.argmax(preds, axis = - 1))

    # convert the true and predicted labels into tensors
    val_true = tf.concat([item for item in y_true], axis = 0)
    true_labels=np.argmax(val_true, axis=1)
    predicted_labels = tf.concat([item for item in y_pred], axis = 0)
    print("test val true and val pred classes: ", val_true[1], predicted_labels[1])

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(num_classes))
    
    # Calculate accuracy from confusion matrix
    cm_accuracy = np.trace(cm) / np.sum(cm)
    print(f'Accuracy calculated from confusion matrix: {cm_accuracy*100:.2f}%')
    class_accuracies = np.diag(cm) / np.sum(cm, axis=1)

    # Plot confusion matrix
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Fold {fold_idx+1} Confusion Matrix')
    plt.savefig(os.path.join(figures_dir, f'fold_{fold_idx+1}_confusion_matrix.png'), dpi=300)
    plt.close()

    return val_accuracy_direct, history, class_accuracies

    
def main():
    seed = 42
    set_seed(seed)
    num_classes = 32
    num_splits = 5
    overwrite = True
    subject_ids = np.arange(1, 25)  # Assuming you have 24 subjects
    all_user_accuracies = {user_id: [] for user_id in range(num_classes)}
    model_save_folder = 'paper_models/category_classifier'
    figures_dir = 'figures/category_classifier'
    os.makedirs(model_save_folder, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    accuracies = []

    # Generate stratified splits for all subjects
    subject_splits = generate_stratified_splits_for_subjects(num_subjects=len(subject_ids), n_splits=num_splits, test_size=0.2)

    for fold_idx in range(num_splits):  # For each fold
        print(f"Training Model {fold_idx + 1}")

        # Create training and validation datasets for the current fold
        train_ds = create_dynamic_dataset(subject_ids, num_classes, fold_idx, subject_splits, is_train=True)
        val_ds = create_dynamic_dataset(subject_ids, num_classes, fold_idx, subject_splits, is_train=False)

        # Inspect datasets before training
        print("Inspecting training dataset batches:")
        inspect_batches(train_ds)
        print("Inspecting validation dataset batches:")
        inspect_batches(val_ds)

        model_save_path = os.path.join(model_save_folder, f'fold_{fold_idx+1}.h5')
        val_accuracy, history, user_accuracies = train_and_evaluate(train_ds, val_ds, fold_idx, figures_dir, model_save_path, num_classes=num_classes, overwrite=overwrite)
        accuracies.append(val_accuracy)

        for user_id, accuracy in enumerate(user_accuracies):
            all_user_accuracies[user_id].append(accuracy)


        if history is not None:
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Fold {fold_idx+1} Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'Fold {fold_idx+1} Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Save the figure
            plt.savefig(f'{figures_dir}/category_classification_fold_{fold_idx+1}_metrics.png')
            plt.close()

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
        # Save results to CSV
    folder_path = 'paper_csvs/category_classifier'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    filepath=f'{folder_path}/metrics_summary_category_classification.csv'
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Category', 'Mean Accuracy', 'Std Accuracy'])
        
        for user_id, accuracies in all_user_accuracies.items():
            if len(accuracies) > 0:  # Ensure there's data to calculate mean/std
                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)
                writer.writerow([user_id + 1, mean_accuracy, std_accuracy])  # Adjust user_id if necessary
            else:
                writer.writerow([user_id + 1, 'N/A', 'N/A'])  # Handle case with no data        
        
if __name__ == "__main__":
    main()
