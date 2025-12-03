import numpy as np
import os
import tensorflow as tf
import random
import csv
import scipy
import itertools
import datetime
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from EEGModels import EEGNet
from tensorflow.keras.mixed_precision import set_global_policy
matplotlib.use('Agg')

def load_data(subject_id, subject_to_label_map, test_size=0.2):
    # Load the preprocessed data
    data_path = f'./database_npy/S{subject_id}_data.npy'
    data = np.load(data_path, mmap_mode='r')  # Use memory-mapping to load data
    
    label = subject_to_label_map[subject_id]
    labels = np.array([label] * data.shape[0])  # Adjust according to your data shape
    labels = tf.keras.utils.to_categorical(labels, num_classes=2)
    
    # Calculate split index for training and validation sets
    split_index = int(data.shape[0] * (1 - test_size))
    
    # Define generator functions for train and validation datasets
    def gen_train():
        for i in range(split_index):
            yield data[i], labels[i]
            
    def gen_val():
        for i in range(split_index, data.shape[0]):
            yield data[i], labels[i]
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_generator(gen_train, output_types=(tf.float32, tf.float32), output_shapes=(data.shape[1:], labels.shape[1:]))
    val_dataset = tf.data.Dataset.from_generator(gen_val, output_types=(tf.float32, tf.float32), output_shapes=(data.shape[1:], labels.shape[1:]))
    
    return train_dataset, val_dataset

def create_datasets(subject_ids, subject_to_label_map, batch_size=256):

    train_datasets = []
    val_datasets = []
    
    for subject_id in subject_ids:
        train_ds, val_ds = load_data(subject_id, subject_to_label_map)
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
    
    # Concatenate datasets of the same class across all subjects
    train_dataset = tf.data.experimental.sample_from_datasets(train_datasets)
    val_dataset = tf.data.experimental.sample_from_datasets(val_datasets)
    
    # Batch and prefetch
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    
    return train_dataset, val_dataset


def create_test_dataset(test_subjects, batch_size=256):
    test_data = []
    test_labels = []
    for subject_id in test_subjects:
        subject_data = np.load(f'./database_npy/S{subject_id}_data.npy')
        # Assuming your subject data is in shape where the first dimension is the sample axis
        subject_labels = np.ones((subject_data.shape[0],), dtype=np.int32)  # All samples labeled as 1 ("denied")
        
        test_data.append(subject_data)
        test_labels.append(subject_labels)
    
    # Concatenate data and labels from all test subjects
    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    # Convert to TensorFlow tensors
    test_data_tensor = tf.convert_to_tensor(test_data, dtype=tf.float32)
    test_labels_tensor = tf.convert_to_tensor(test_labels, dtype=tf.int32)
    test_labels_tensor = tf.keras.utils.to_categorical(test_labels_tensor, num_classes=2)  # Convert labels to one-hot encoding
    
    # Create a TensorFlow dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data_tensor, test_labels_tensor))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return test_dataset


def compute_weights_for_classes(subject_ids, allowance_number, is_open_set):
    total_subjects = len(subject_ids)
    if is_open_set == False:
        # for closed set we use all users for training and so if we have 6 allowed subjects and 18 denied subjects we have an inbalance
        labels = np.array([0] * allowance_number + [1] * (total_subjects - allowance_number))
    else:
        # for open set, we always have an equal number of users in the allow and deny group
        labels = np.array([0] * allowance_number + [1] * (allowance_number))
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weight_dict = dict(enumerate(class_weights))
    return class_weight_dict

def set_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)


def create_random_splits(subject_ids, k, allowance_number):
    """
    Create random splits of subjects into two classes for k folds, based on the allowance_number.
    Class 0 will have allowance_number subjects, and Class 1 will have the rest.
    
    Parameters:
    - subject_ids: Array or list of subject IDs.
    - k: Number of folds.
    - allowance_number: Number of subjects in Class 0.
    - seed: Random seed for reproducibility.
    
    Returns:
    A generator yielding tuples of (class_0_subjects, class_1_subjects) for each fold.
    """
    all_subjects = np.random.permutation(subject_ids)

    # Calculate the number of subjects not included in the allowance (Class 1)
    remainder = len(subject_ids) - allowance_number

    for _ in range(k):
        np.random.shuffle(all_subjects)
        
        # Determine the subjects in each class for this split
        class_0_subjects = all_subjects[:allowance_number]
        class_1_subjects = all_subjects[allowance_number:allowance_number+remainder]
        
        yield class_0_subjects, class_1_subjects

def train_and_evaluate(train_dataset, val_dataset, fold_number, class_weight_dict, is_open_set, allowance_number):
    model = EEGNet(nb_classes=2, Chans=33, Samples=1500, dropoutRate=0.5, kernLength=256, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout')
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    if is_open_set:
        log_dir = f"./paper_logs/authentication_open_set_{allowance_number}_{allowance_number}_model_fold_{fold_number+1}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_folder = f'paper_models/authentication/authentication_open_set_{allowance_number}_{allowance_number}'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_path = f'{model_folder}/authentication_open_set_{allowance_number}_{allowance_number}_model_fold_{fold_number+1}.h5'
    else:
        log_dir = f"./paper_logs/authentication/authentication_closed_set_{allowance_number}_{24-allowance_number}_model_fold_{fold_number+1}" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_folder =  f'paper_models/authentication_closed_set_{allowance_number}_{24-allowance_number}'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_path = f'{model_folder}/authentication_closed_set_{allowance_number}_{24-allowance_number}_model_fold_{fold_number+1}.h5'

    def get_tensorboard_callback(log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        return tensorboard_callback
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, verbose=1),
        get_tensorboard_callback(log_dir)
    ]

    history = model.fit(train_dataset, epochs=200, validation_data=val_dataset, callbacks=callbacks, class_weight=class_weight_dict, verbose=2)

    val_data, val_labels = next(iter(val_dataset.unbatch().batch(10000)))
    val_labels_pred = model.predict(val_data)
    val_labels_pred = np.argmax(val_labels_pred, axis=1)
    val_labels_true = np.argmax(val_labels.numpy(), axis=1)

    accuracy = accuracy_score(val_labels_true, val_labels_pred)
    precision = precision_score(val_labels_true, val_labels_pred, average='macro')
    recall = recall_score(val_labels_true, val_labels_pred, average='macro')
    f1 = f1_score(val_labels_true, val_labels_pred, average='macro')

    return model, history, accuracy, precision, recall, f1


def save_metrics_summary_to_csv(accuracies, precisions, recalls, f1_scores, test_accuracies, test_precisions, test_recalls, test_f1_scores, allowance_number, is_open_set):
    if is_open_set:
        csv_path = f"paper_csvs/metrics_summary_authentication_open_set_{allowance_number}_{allowance_number}.csv"
    else:
        csv_path = f"paper_csvs/metrics_summary_authentication_closed_set_{allowance_number}_{24-allowance_number}.csv"
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Define header based on whether it is an open-set scenario
        if is_open_set:
            headers = ["Metric", "Training/Validation Mean", "Training/Validation Std", "Test Mean", "Test Std"]
        else:
            headers = ["Metric", "Mean", "Std"]
        writer.writerow(headers)
        
        metrics = [("Accuracy", accuracies, test_accuracies),
                   ("Precision", precisions, test_precisions),
                   ("Recall", recalls, test_recalls),
                   ("F1 Score", f1_scores, test_f1_scores)]
        
        for metric_name, train_val_scores, test_scores in metrics:
            train_val_mean = np.mean(train_val_scores)
            train_val_std = np.std(train_val_scores)
            
            if is_open_set:
                test_mean = np.mean(test_scores)
                test_std = np.std(test_scores)
                writer.writerow([metric_name, f"{train_val_mean:.4f}", f"{train_val_std:.4f}", f"{test_mean:.4f}", f"{test_std:.4f}"])
            else:
                writer.writerow([metric_name, f"{train_val_mean:.4f}", f"{train_val_std:.4f}"])


def evaluate_model(test_dataset, model, threshold=0.5):  # Adjust the threshold as needed
    # Predict on the test dataset

    y_pred_probs = model.predict(test_dataset)
    print("predicted probabilities: ", y_pred_probs)
    y_pred_classes = (y_pred_probs[:, 1] >= threshold).astype(int)  # Assuming class 1 is "denied"
    
    # Extract true labels from the dataset
    y_true = np.concatenate([y for _, y in test_dataset], axis=0)
    y_true_classes = np.argmax(y_true, axis=1)
    
    # Compute metrics
    test_accuracy = accuracy_score(y_true_classes, y_pred_classes)
    test_precision = precision_score(y_true_classes, y_pred_classes, average='macro')
    test_recall = recall_score(y_true_classes, y_pred_classes, average='macro')
    test_f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
    
    return test_accuracy, test_precision, test_recall, test_f1

def inspect_batches(ds, num_batches=1000):
    for batch_index, (x, y) in enumerate(ds.take(num_batches)):
        labels_indices = tf.argmax(y, axis=1).numpy()
        unique, counts = np.unique(labels_indices, return_counts=True)
        print(f"Batch {batch_index + 1}: Unique labels - {unique}, Counts - {counts}")

# Assuming the rest of your necessary imports and function definitions (e.g., EEGNet, train_and_evaluate, create_datasets) are here

def main():
    seed = 42
    set_global_policy('mixed_float16')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    set_seed(seed)

    allowance_number = 12  # Number of subjects considered "authenticated"
    is_open_set = False  # Flag to control the experiment type
    batch_size = 256  # Batch size for training
    k = 5  # Number of folds
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    test_accuracies, test_precisions, test_recalls, test_f1_scores = [], [], [], []

    for fold_number in range(k):
        subject_ids = np.arange(1, 25)  # Reset all subject IDs for each fold
        np.random.shuffle(subject_ids)  # Shuffle to ensure randomness for each fold
        subject_to_label_map = {}

        if is_open_set:
            allowed_subjects = subject_ids[:allowance_number]
            denied_subjects = subject_ids[allowance_number:allowance_number*2]
            training_subjects = np.concatenate((allowed_subjects, denied_subjects))
            test_subjects = subject_ids[allowance_number*2:]
            for subject in allowed_subjects:
                subject_to_label_map[subject] = 0
            for subject in denied_subjects:
                subject_to_label_map[subject] = 1
        else:
            training_subjects = subject_ids
            for idx, subject in enumerate(training_subjects):
                if idx < allowance_number:
                    subject_to_label_map[subject] = 0
                else:
                    subject_to_label_map[subject] = 1
            test_subjects = []

        class_weight_dict = compute_weights_for_classes(training_subjects, allowance_number, is_open_set)

        print("The subject to label map is: ", subject_to_label_map)
        print("The class weights are: ", class_weight_dict)

        # Create datasets for training and validation.
        train_dataset, val_dataset = create_datasets(training_subjects, subject_to_label_map, batch_size)
        print("Inspecting training dataset batches:")
        inspect_batches(train_dataset)
        print("Inspecting validation dataset batches:")
        inspect_batches(val_dataset)

        # Train and evaluate using the aggregated datasets
        model, history, accuracy, precision, recall, f1 = train_and_evaluate(train_dataset, val_dataset, fold_number, class_weight_dict, is_open_set, allowance_number)
        # Collect and store metrics
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # Ensure the figures directory exists
        if is_open_set:
            figures_dir = f'./figures/authentication_open_set_{allowance_number}_{allowance_number}'
            if not os.path.exists(figures_dir):
                os.makedirs(figures_dir)
        else:
            figures_dir = f'./figures/authentication_closed_set_{allowance_number}_{24-allowance_number}'
            if not os.path.exists(figures_dir):
                os.makedirs(figures_dir)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Fold {fold_number+1} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Fold {fold_number+1} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        
        # Save the figure
        plt.savefig(f'{figures_dir}/authentication_fold_{fold_number+1}_metrics.png')
        plt.close()
    
        if is_open_set:
            threshold = 0.2
            test_dataset = create_test_dataset(test_subjects, batch_size)
            test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(test_dataset, model, threshold=threshold)
            
            test_accuracies.append(test_accuracy)
            test_precisions.append(test_precision)
            test_recalls.append(test_recall)
            test_f1_scores.append(test_f1)

    
    save_metrics_summary_to_csv(accuracies, precisions, recalls, f1_scores, test_accuracies, test_precisions, test_recalls, test_f1_scores, allowance_number, is_open_set)
    print(f"Saved metrics summary to CSV for allowance number {allowance_number}.")


if __name__ == "__main__":
    main()

