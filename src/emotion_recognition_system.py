import os
import sys
from os.path import join

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from constants import *
from dataset_loader import DatasetLoader


class EmotionRecognition:
    def __init__(self):
        self.dataset = DatasetLoader()

    def build_network(self):
        print('[+] Building CNN')
        self.network = input_data(shape=[None, SIZE_FACE, SIZE_FACE, 1])
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 128, 4, activation='relu')
        self.network = dropout(self.network, 0.3)
        self.network = fully_connected(self.network, 3072, activation='relu')
        self.network = fully_connected(self.network, len(EMOTIONS), activation='softmax')
        self.network = regression(self.network, optimizer='momentum', loss='categorical_crossentropy')

        self.model = tflearn.DNN(self.network, checkpoint_path=DATA_SET_DIR + '/emotion_recognition', max_checkpoints=1,
                                 tensorboard_verbose=2)
        self.load_model()

    def load_saved_dataset(self):
        files = []
        for x in os.listdir(DATA_SET_DIR):
            files.append(os.path.splitext(x)[0])
        if files.__contains__(SAVED_DATA_SET):
            self.dataset.load_from_save()
            print('[+] Dataset found and loaded')
        else:
            print("Training Set not found \nCreate New Now?")
            choice = input("Enter Your Choice (Y/N)")
            if (choice == 'Y') | (choice == 'y'):
                import csv_to_numpy
                self.start_training()
            else:
                exit(0)

    def load_model(self):
        files = []
        for x in os.listdir(DATA_SET_DIR):
            files.append(os.path.splitext(x)[0])
        print(files)
        if files.__contains__(SAVE_MODEL_FILENAME):
            self.model.load(join(DATA_SET_DIR, SAVE_MODEL_FILENAME))
            print('[+] Model loaded from ' + SAVE_MODEL_FILENAME)
        else:
            print('[+] Model Not loaded. File named' + SAVE_MODEL_FILENAME + ' Not Exist')

    def start_training(self):
        self.load_saved_dataset()
        self.build_network()
        if self.dataset is None:
            self.load_saved_dataset()
        # Training
        print('[+] Training network')
        # self.model.fit(self.dataset.images, self.dataset.labels,
        #                validation_set=(self.dataset.images_test, self.dataset.labels_test), n_epoch=100,
        #                batch_size=50, shuffle=True, show_metric=True, snapshot_step=200, snapshot_epoch=True,
        #                run_id='emotion_recognition'
        #                )

        self.model.fit(self.dataset.images, self.dataset.labels,
                       validation_set=(self.dataset.images_test, self.dataset.labels_test), n_epoch=2, batch_size=50,
                       shuffle=True, show_metric=True, snapshot_step=200, snapshot_epoch=True,
                       run_id='emotion_recognition'
                       )

    def predict(self, image):
        if image is None:
            return None
        image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
        return self.model.predict(image)

    def save_model(self):
        self.model.save(join(DATA_SET_DIR, SAVE_MODEL_FILENAME))
        print('[+] Model trained and saved at ' + SAVE_MODEL_FILENAME)


def show_usage():
    # I din't want to have more dependecies
    print('[!] Usage: python emotion_recognition_system.py')
    print('\t emotion_recognition_system.py train \t Trains and saves model with saved dataset')
    print('\t emotion_recognition_system.py run \t Launch the Demonstration of concept')


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        show_usage()
        exit()

    network = EmotionRecognition()
    if sys.argv[1] == 'train':
        network.start_training()
        network.save_model()
    elif sys.argv[1] == 'run':
        import run_ERS
    else:
        show_usage()
