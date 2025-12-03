Public EEG database can be found at: [OneDrive](https://ctipub-my.sharepoint.com/:f:/g/personal/dan_curavale_stud_etti_upb_ro/Ej_dS3KWqiNDuK8lT9qT4lwBwQxUbhgkNYiQBfJ8VM-TkA?e=VEQnBB). Data is already preprocessed as presented in subsection "Data preprocessing" of the paper.

After adding the Personal Hobbies Database to the working directory:
1.  Run the "personalized_emotion_classifier.py" script to train the emotion classification model presented in subsection A of the "Experimental scenarios and results" section of the paper.
2.  Run the "category_classifier.py" script to train the category classification model presented in subsection B of the same section.
3. Run the "authenticate_binary_classifier.py" script to train the binary classification model presented in subsection C of the same section.
4. Run the "identity_classifier.py" script to train the user identification model presented in subsection D.

"EEGModels.py" contains the architecture of the [EEGNet](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta) neural network used in our work.