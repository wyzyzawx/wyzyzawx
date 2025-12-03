import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
import itertools
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from EEGModels import EEGNet
from tensorflow.keras.mixed_precision import set_global_policy
matplotlib.use('Agg')

set_global_policy('mixed_float16')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def load_data_on_demand(subject_id, is_train, fold_idx, n_splits, num_classes):
    """
    A generator function that loads data for a specific subject on demand, 
    yielding data for either training or validation based on the fold index.
    """
    # Load subject data path
    data_path = f'./database_npy/S{subject_id}_data.npy'
    data = np.load(data_path, mmap_mode='r')

    labels = np.full(shape=data.shape[0], fill_value=subject_id - 1)
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    # Calculate the total size and validation split size
    total_size = len(data)
    val_split_size = total_size // n_splits

    # Calculate start and end indices for validation split based on the fold index
    val_start_idx = fold_idx * val_split_size
    val_end_idx = val_start_idx + val_split_size if (fold_idx < n_splits - 1) else total_size

    if is_train:
        # For training, use all data except the validation split
        indices = itertools.chain(range(0, val_start_idx), range(val_end_idx, total_size))
    else:
        # For validation, use just the validation split
        indices = range(val_start_idx, val_end_idx)

    # Iterate over the correct set of indices and yield data and labels
    for i in indices:
        yield tf.expand_dims(data[i], axis=-1), labels[i]


def create_dynamic_dataset(subject_ids, num_classes, fold_idx, n_splits, is_train):
    datasets = []
    for subject_id in subject_ids:
        # Here, we bind the current loop variable to a new mandatory argument of the generator function.
        gen_func = lambda subject_id=subject_id: load_data_on_demand(subject_id, is_train, fold_idx, n_splits, num_classes)
        ds = tf.data.Dataset.from_generator(
            gen_func,
            output_types=(tf.float32, tf.float32),
            output_shapes=((33, 1500, 1), (num_classes,))
        )
        datasets.append(ds)

    combined_ds = datasets[0]
    for ds in datasets[1:]:
        combined_ds = combined_ds.concatenate(ds)

    return combined_ds.shuffle(10000).batch(256).prefetch(tf.data.AUTOTUNE)


def train_and_evaluate(train_ds, val_ds, fold_idx, figures_dir, model_save_path, num_classes=24, overwrite=False):
    if os.path.exists(model_save_path) and overwrite==False:
        print(f"Model already exists at {model_save_path}. Loading model...")
        model = tf.keras.models.load_model(model_save_path)
        history = None
    else:
        log_dir = f"./paper_logs/identity_classifier/model_fold_{fold_idx+1}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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

def inspect_batches(ds, num_batches=1000):
    for batch_index, (x, y) in enumerate(ds.take(num_batches)):
        labels_indices = tf.argmax(y, axis=1).numpy()
        unique, counts = np.unique(labels_indices, return_counts=True)
        print(f"Batch {batch_index + 1}: Unique labels - {unique}, Counts - {counts}")

def set_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)


def main():
    seed=42
    set_seed(seed)
    num_classes = 24
    num_splits = 5
    overwrite=False
    subject_ids = np.arange(1, num_classes + 1)
    all_user_accuracies = {user_id: [] for user_id in range(num_classes)}
    model_save_folder = 'paper_models/identity_classifier'
    figures_dir = 'figures/identity_classifier'
    os.makedirs(model_save_folder, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    accuracies = []


    for fold_idx in range(num_splits):
        # Create dynamic datasets for the current fold
        train_ds = create_dynamic_dataset(subject_ids, num_classes, fold_idx, num_splits, is_train=True)
        val_ds = create_dynamic_dataset(subject_ids, num_classes, fold_idx, num_splits, is_train=False)

        # Call the inspection function before starting the training to check the first few batches
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
            plt.savefig(f'{figures_dir}/identification_fold_{fold_idx+1}_metrics.png')
            plt.close()

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
        # Save results to CSV
    folder_path = 'paper_csvs/identity_classifier'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    filepath=f'{folder_path}/metrics_summary_identification.csv'
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['User ID', 'Mean Accuracy', 'Std Accuracy'])
        
        for user_id, accuracies in all_user_accuracies.items():
            if len(accuracies) > 0:  # Ensure there's data to calculate mean/std
                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)
                writer.writerow([user_id + 1, mean_accuracy, std_accuracy])  # Adjust user_id if necessary
            else:
                writer.writerow([user_id + 1, 'N/A', 'N/A'])  # Handle case with no data        

if __name__ == "__main__":
    main()






















exit(0)
from utils import *

data = np.empty((0, 33, 1500))
labels = []
x_train_concat = np.empty((0, 33, 1500))
y_train_concat = np.empty(0)
x_test_concat = np.empty((0, 33, 1500))
y_test_concat = np.empty(0)
count = 0

lst = [*range(1, 26, 1)]
for k in lst:

    data_user, labels_user = ExtractIdentificationData(k)


    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data_user, labels_user, test_size=0.2,
                                                                                train_size=0.8, random_state=42,
                                                                                shuffle=True, stratify=labels_user)


    x_train_concat = np.append(x_train_concat, x_train, axis=0)
    y_train_concat = np.append(y_train_concat, y_train, axis=0)
    x_test_concat = np.append(x_test_concat, x_test, axis=0)
    y_test_concat = np.append(y_test_concat, y_test, axis=0)
    labels = np.concatenate((labels, labels_user))
    print(k, 'subiect curent')
    print(labels_user[1:10])

    count += 1


y_train_concat = to_categorical(y_train_concat)
y_test_concat = to_categorical(y_test_concat)


nb_classes = 25

model = EEGNet(nb_classes, Chans=33, Samples=1500, dropoutRate=0.5, kernLength=256, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
save_model_path = '.\\Modele_Licenta\\identity_classifier_25_users_final.h5'
history = model.fit(x_train_concat, y_train_concat, batch_size=32, epochs=1000, verbose=2, callbacks=[get_tensorboard_callback(),
                     tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=200, verbose=1, mode="auto", baseline=None, restore_best_weights=False),
                     ModelCheckpoint(save_model_path, save_best_only=True, verbose=1)], validation_data=(x_test_concat, y_test_concat))

