import numpy as np
import tensorflow as tf
import scipy

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

def extract_data(subject):
    data_mat = scipy.io.loadmat('Official_Preprocessed_Data/S' + str(subject) + '_data.mat')
    category_order = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 20, 21, 22, 23, 24, 25, 26, 27, 28, 2, 30, 31, 32,
                      3, 4, 5, 6, 7, 8, 9, 29]
    if subject == 4:
        del category_order[23]
    if subject == 13:
        del category_order[0]
    if subject == 14:
        del category_order[0]
        del category_order[1]
    if subject == 16:
        del category_order[21]
    if subject == 20:
        del category_order[27]
        del category_order[27]
    if subject == 22:
        del category_order[23]

    for key, value in data_mat.items():
        if key == 'data':
            data = value

    data = data.squeeze()
    matrix_current = np.empty((0, 33, 1500))
    j = []
    for i in range(0, len(data)):
        data[i] = np.swapaxes(data[i], 0, 2)
        data[i] = np.swapaxes(data[i], 1, 2)
        j.append(len(data[i]))
        matrix_current = np.concatenate((matrix_current, data[i][:, :, 500:2000]), axis=0)
    labels_aux = []

    for i in range(len(category_order)):
        for _ in range(j[i]):
            labels_aux.append(category_order[i])

    return matrix_current, np.array(labels_aux)

def main():
    subject_ids = range(1, 25)
    # De la subiectul 4 (subiectul 5 in chestionar) a fost eliminata categoria 24 (UNI)
    # De la subiectul 13 (subiectul 14 in chestionar) a fost eliminata categoria 1 
    # De la subiectul 14 (subiectul 15 in chestionar) a fost eliminata categoria 1 si 3
    # De la subiectul 16 (subiectul 16 in chestionar) a fost eliminata categoria 22
    # De la subiectul 20 (subiectul 21 in chestionar) a fost eliminata categoria 28 si 29
    # De la subiectul 22 (subiectul 23 in chestionar) a fost eliminata categoria 24 (UNI)
    for subject in subject_ids:
        data2, labels_category = extract_data(subject)
        data, labels = get_data(subject)
        np.save(f'database_npy_updated/S{subject}_data', data)
        np.save(f'database_npy_updated/S{subject}_likeness_labels', labels)
        np.save(f'database_npy_updated/S{subject}_category_labels', labels_category)
        print(np.allclose(data,data2))
        print(f'data shape for subject {subject} is {data.shape}')
        print(f'likeness labels len for subject {subject} is {len(labels)}')
        print(f'category labels len for subject {subject} is {len(labels_category)}')



if __name__ == "__main__":
    main()
