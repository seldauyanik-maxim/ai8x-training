###################################################################################################
#
# Copyright (C) 2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Classes and functions used to create SMS Motor Data Dataset
"""

import os
import pickle
import sys
import math
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from numpy.fft import fft
import pandas as pd
import torch

import ai8x
from utils import data_loader_utils
from scipy import integrate


def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))
    
class SMSMotorData(Dataset):

    """
    Data Loader class for the Motor Data

    Each sequence was generated at a 4 kHz sampling rate
    NOTE: pre-processed npy file is utilized. See related notebook for details
    """
    raw_data_sampling_rate_Hz = 4000
    cnn_1dinput_len = 512

    # 80% of th enormal samples are separated into training set
    train_ratio = 0.8

    #TODO: Move all static methods to motordata_preprocessing.py

    @staticmethod
    def sliding_windows_1d(array, window_size, overlap_ratio):

        window_overlap = math.ceil(window_size * overlap_ratio)

        slide_amount = window_size - window_overlap
        num_of_windows = math.floor((len(array) - window_size) / slide_amount) + 1

        result_list = np.zeros((num_of_windows, window_size))

        for i in range(num_of_windows):
            start_idx = slide_amount * i
            end_idx = start_idx + window_size
            result_list[i] = array[start_idx:end_idx]

        return result_list
    
    @staticmethod
    def sliding_windows_on_columns_of_2d(array, window_size, overlap_ratio):

        array_len, num_of_cols = array.shape

        window_overlap = math.ceil(window_size * overlap_ratio)
        slide_amount = window_size - window_overlap
        num_of_windows = math.floor((array_len - window_size) / slide_amount) + 1

        result_list = np.zeros((num_of_cols, num_of_windows, window_size))

        for i in range(num_of_cols):
            result_list[i, :, :] = SMSMotorData.sliding_windows_1d(array[:, i], window_size, overlap_ratio)

        return result_list

    @staticmethod
    def split_file_raw_data(file_raw_data, file_raw_data_fs_in_Hz, duration_in_sec, overlap_ratio):

        num_of_samples_per_window = int(file_raw_data_fs_in_Hz * duration_in_sec)
        
        sliding_windows = SMSMotorData.sliding_windows_on_columns_of_2d(file_raw_data,
                                                                    num_of_samples_per_window,
                                                                    overlap_ratio)
        return sliding_windows

    def process_file_and_return_signal_windows(self, file_raw_data):
 
        # Downsample signal first if not specified
        if not self.ds_after_fft:
            new_sampling_rate = int(SMSMotorData.raw_data_sampling_rate_Hz / self.downsampling_ratio)
            file_raw_data_sampled = file_raw_data[::self.downsampling_ratio,:]
        else:
            new_sampling_rate = SMSMotorData.raw_data_sampling_rate_Hz
            file_raw_data_sampled = file_raw_data

        file_raw_data_windows = SMSMotorData.split_file_raw_data(file_raw_data_sampled,
                                                             new_sampling_rate,
                                                             self.signal_duration_in_sec,
                                                             self.overlap_ratio)

        # First dimension: 3
        # Second dimension: number of windows
        # Third dimension: Window for self.duration_in_sec
        num_features = file_raw_data_windows.shape[0]
        num_windows = file_raw_data_windows.shape[1]
        input_window_size = file_raw_data_windows.shape[2]

        if self.ds_after_fft:
            fft_input_window_size = next_power_of_2(input_window_size / self.downsampling_ratio) * self.downsampling_ratio
        else:
            fft_input_window_size = next_power_of_2(input_window_size)

        fft_output_window_size = int(fft_input_window_size / 2) 

        if((not self.ds_after_fft and fft_output_window_size != SMSMotorData.cnn_1dinput_len) or 
           (self.ds_after_fft and fft_output_window_size != SMSMotorData.cnn_1dinput_len * self.downsampling_ratio)):
            raise ValueError(f"Please adjust downsampling_ratio (current: {self.downsampling_ratio}) and signal_duration (current:{self.signal_duration_in_sec}) such that resulting window after fft and ds operations has cnn_input_size: {SMSMotorData.cnn_1dinput_len} samples")

        file_cnn_signals = np.zeros((num_features, num_windows, SMSMotorData.cnn_1dinput_len))

        # Perform FFT on each window () for each feature
        for feature in range(num_features):
            for window in range(num_windows):

                signal_for_fft = file_raw_data_windows[feature, window, :]

                if self.dc_removed:
                    signal_for_fft -= np.mean(signal_for_fft)
                
                #fft_out = abs(fft(signal_for_fft, n=fft_input_window_size))

                fft_out = abs(fft(signal_for_fft))
                fft_out = fft_out[:fft_output_window_size] 

                # Downsample after fft if specified
                if self.ds_after_fft:
                    fft_out = fft_out[::self.downsampling_ratio]

                file_cnn_signals[feature, window, :] = fft_out

        # Reshape from (num_features, num_windows, window_size) 
        #        into: (num_windows, num_features, window_size)
        file_cnn_signals = file_cnn_signals.transpose([1, 0, 2])
        return file_cnn_signals

    def __init__(self, root, d_type, transform=None,
                 anomaly_label=False,
                 downsampling_ratio=2,
                 signal_duration_in_sec=0.5,
                 overlap_ratio=0.75, 
                 dc_removed=True,
                 dbu_ind=False,
                 ds_after_fft=False,
                 norm_per_ch=False):

        if d_type not in ('test', 'train'):
            raise ValueError("d_type can only be set to 'test' or 'train'")

        if not isinstance(downsampling_ratio, int) or downsampling_ratio < 1:
            raise ValueError("downsampling_ratio can only be set to an integer value greater than 0")

        number_of_raw_data_samples_per_window = signal_duration_in_sec * int(SMSMotorData.raw_data_sampling_rate_Hz / downsampling_ratio)

        if next_power_of_2(number_of_raw_data_samples_per_window) != SMSMotorData.cnn_1dinput_len * 2:
            raise ValueError("Please adjust downsampling_ratio and signal_duration such that resulting window has 1024 samples (cnn_input_size * 2)")

        if not dbu_ind:
            self.info_dataframe_pkl = os.path.join(root, self.__class__.__name__, 'structured_smsmotordata_dataframe.pkl')
        else:
            self.info_dataframe_pkl = os.path.join(root, self.__class__.__name__, 'structured_sms_dataframe_dbu_induction.pkl')

        if not os.path.exists(self.info_dataframe_pkl):
            print('\nPlease acquire preprocessed dat ainformation dataframe pickle file')
            print('Place pickle file to path [data_dir]/SMSMotorData/.')

            sys.exit()

        self.d_type = d_type
        self.transform = transform
        self.anomaly_label = anomaly_label
    
        self.downsampling_ratio = downsampling_ratio
        self.signal_duration_in_sec = signal_duration_in_sec
        self.overlap_ratio = overlap_ratio
        self.dbu_ind = dbu_ind
        self.ds_after_fft = ds_after_fft
        self.dc_removed = dc_removed
        self.norm_per_ch=norm_per_ch

        self.num_of_features = 3
        
        processed_folder = \
            os.path.join(root, self.__class__.__name__, 'processed')

        # if dc_removed:
        #     processed_folder = \
        #         os.path.join(root, self.__class__.__name__, 'processed')
        # else:
        #     processed_folder = \
        #         os.path.join(root, self.__class__.__name__, 'old_processed')
        
        data_loader_utils.makedir_exist_ok(processed_folder)

        if self.dbu_ind:
            specs_identifier = f'anomaly_label_{self.anomaly_label}_' + \
                            f'ds_ratio_{self.downsampling_ratio}_' + \
                            f'signal_dur_{self.signal_duration_in_sec}_' + \
                            f'overlap_ratio_{self.overlap_ratio}_' +\
                            f'dbu_ind_True'
        else:
            specs_identifier = f'anomaly_label_{self.anomaly_label}_' + \
                            f'ds_ratio_{self.downsampling_ratio}_' + \
                            f'signal_dur_{self.signal_duration_in_sec}_' + \
                            f'overlap_ratio_{self.overlap_ratio}'

        if self.dc_removed:
            specs_identifier += '_dc_removed'

        if self.ds_after_fft:
            specs_identifier += '_ds_after_fft'

        if self.norm_per_ch:
            specs_identifier += '_norm_per_ch'

        train_dataset_pkl_file_path = \
            os.path.join(processed_folder, f'train_{specs_identifier}.pkl')

        test_dataset_pkl_file_path =  \
            os.path.join(processed_folder, f'test_{specs_identifier}.pkl')

        if self.d_type == 'train':
            self.dataset_pkl_file_path = train_dataset_pkl_file_path

        elif self.d_type == 'test':
            self.dataset_pkl_file_path = test_dataset_pkl_file_path

        self.signal_list = []
        self.lbl_list = []

        self.__create_pkl_files()
        self.is_truncated = False

    def __create_pkl_files(self):
        if os.path.exists(self.dataset_pkl_file_path):

            print('\nPickle files are already generated ...\n')

            (self.signal_list, self.lbl_list) = \
                pickle.load(open(self.dataset_pkl_file_path, 'rb'))
            return

        self.__gen_datasets()

    def __gen_datasets(self):
        print('\nGenerating dataset pickle files from the raw data files...\n')

        df = pd.read_pickle(self.info_dataframe_pkl)
        df_normals = df[df['health_status'] == 0]
        df_anormals = df[df['health_status'] != 0]

        # TODO: may use lambdas and map for faster iteration instead of iterrows
        # LOAD NORMAL FEATURES
        normal_features = list()
        for _, row in df_normals.iterrows():
            raw_data = row['data'][0, :].transpose([1, 0])
            cnn_signals = self.process_file_and_return_signal_windows(raw_data)
            for i in range(cnn_signals.shape[0]):
                normal_features.append(cnn_signals[i])
    
        normal_features = np.asarray(normal_features)

        if self.norm_per_ch:
            # Keep min/max values using normal data:
            self.normal_features_max = np.max(normal_features, axis=(0, 2)) # (3)
            self.normal_features_min = np.min(normal_features, axis=(0, 2)) # (3)

            # Normalize normal data:
            for feature in range(self.num_of_features):
                normal_features[:, feature, :] = \
                    (normal_features[:, feature, :] - self.normal_features_min[feature]) / \
                    (self.normal_features_max[feature] - self.normal_features_min[feature])
        else:
            # TODO: Can also use min-max scaler from sklearn
            self.normal_features_max = np.max(normal_features, axis=0)
            self.normal_features_min = np.min(normal_features, axis=0) # (3, 512)

            # Normalize normal data:
            for feature in range(self.num_of_features):
                for signal in range(self.cnn_1dinput_len):
                    normal_features[:, feature, signal] = \
                        (normal_features[:, feature, signal] - self.normal_features_min[feature, signal]) / \
                        (self.normal_features_max[feature, signal] - self.normal_features_min[feature, signal])

        # Shuffles only in first dimension as intended
        np.random.shuffle(normal_features)

        num_training = int(SMSMotorData.train_ratio * len(normal_features))

        train_features = normal_features[0:num_training]
        test_normal_features = normal_features[num_training:]

        # TODO: may use lambdas and map for faster iteration instead of iterrows
        # LOAD ANORMAL FEATURES
        anomaly_features = list()
        for _, row in df_anormals.iterrows():
            raw_data = row['data'][0, :].transpose([1, 0])
            cnn_signals = self.process_file_and_return_signal_windows(raw_data)
            for i in range(cnn_signals.shape[0]):
                anomaly_features.append(cnn_signals[i])

        anomaly_features = np.asarray(anomaly_features)

        if self.norm_per_ch:
            # Normalize anomaly data:
            for feature in range(self.num_of_features):
                anomaly_features[:, feature, :] = \
                    (anomaly_features[:, feature, :] - self.normal_features_min[feature]) / \
                    (self.normal_features_max[feature] - self.normal_features_min[feature])
        else:
            # Normalize normal data:
            for feature in range(self.num_of_features):
                for signal in range(self.cnn_1dinput_len):
                    anomaly_features[:, feature, signal] = \
                        (anomaly_features[:, feature, signal] - self.normal_features_min[feature, signal]) / \
                        (self.normal_features_max[feature, signal] - self.normal_features_min[feature, signal])

        # Shuffles only in first dimension as intended
        np.random.shuffle(anomaly_features)

        # ARRANGE TEST-TRAIN SPLIT AND LABELS
        if self.d_type == 'train':
            self.lbl_list = [train_features[i, :, :] for i in range(train_features.shape[0])]
            self.signal_list = [torch.Tensor(label) for label in self.lbl_list]
            self.lbl_list = list(self.signal_list)

            if self.anomaly_label:
                self.lbl_list = np.zeros([len(self.signal_list), 1])

        elif self.d_type == 'test':

            # Testing in training phase includes only normal test samples
            if not self.anomaly_label:
                test_data = test_normal_features
            else:
                test_data = np.concatenate((test_normal_features, anomaly_features), axis=0)

            self.lbl_list = [test_data[i, :, :] for i in range(test_data.shape[0])]
            self.signal_list = [torch.Tensor(label) for label in self.lbl_list]
            self.lbl_list = list(self.signal_list)

            if self.anomaly_label:
                self.lbl_list = np.concatenate(
                                    (np.zeros([len(test_normal_features), 1]),
                                     np.ones([len(anomaly_features), 1])), axis=0)

        # Save pickle file
        pickle.dump((self.signal_list, self.lbl_list), open(self.dataset_pkl_file_path, 'wb'))

    def __len__(self):
        if self.is_truncated:
            return 1
        return len(self.signal_list)

    def __getitem__(self, index):

        if index >= len(self):
            raise IndexError

        if self.is_truncated:
            index = 0

        signal = self.signal_list[index]
        lbl = self.lbl_list[index]

        if self.transform is not None:
            signal = self.transform(signal)

            if not self.anomaly_label:
                lbl = self.transform(lbl)

        if self.anomaly_label:
            lbl = lbl.astype(np.long)
        else:
            lbl = lbl.numpy().astype(np.float32)

        return signal, lbl


def smsmotordata_get_datasets(data, load_train=True, load_test=True,
                          anomaly_label=False,
                          downsampling_ratio=2, 
                          signal_duration_in_sec=0.5,
                          overlap_ratio=0.75,
                          dc_removed=True,
                          dbu_ind=False,
                          ds_after_fft=False,
                          norm_per_ch=False):
    """
    Returns SMSMotorData datasets
    """
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])

        train_dataset = SMSMotorData(root=data_dir, d_type='train',
                                 transform=train_transform,
                                 anomaly_label=anomaly_label,
                                 downsampling_ratio = downsampling_ratio,
                                 signal_duration_in_sec=signal_duration_in_sec,
                                 overlap_ratio=overlap_ratio,
                                 dc_removed=dc_removed,
                                 dbu_ind=dbu_ind,
                                 ds_after_fft=ds_after_fft,
                                 norm_per_ch=norm_per_ch)

        print(f'Train dataset length: {len(train_dataset)}\n')
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])

        test_dataset = SMSMotorData(root=data_dir, d_type='test',
                                 transform=test_transform,
                                 anomaly_label=anomaly_label,
                                 downsampling_ratio = downsampling_ratio,
                                 signal_duration_in_sec=signal_duration_in_sec,
                                 overlap_ratio=overlap_ratio,
                                 dc_removed=dc_removed,
                                 dbu_ind=dbu_ind,
                                 ds_after_fft=ds_after_fft,
                                 norm_per_ch=norm_per_ch)

        print(f'Test dataset length: {len(test_dataset)}\n')
    else:
        test_dataset = None

    return train_dataset, test_dataset


def smsmotordata_dur_0_5_overlap_0_75_get_datasets_for_train(data, load_train=True, load_test=True, dc_removed=True, dbu_ind=False):
    """
    Returns datasets for training purposes: labels are equivalent to signals
    Note: For training, only normal samples are used, test set here is therefore
          part of original set of normal samples (ratio: SMSMotorData.train_ratio).
    """
    # NOTE: Signal duration can be modified.
    #       But downsampling ratio is arranged such that
    #       Each signal window contains exactly 1000 (or 1024) samples
    #       (As after FFT, half of this window will be fed to CNN & CNN input size should be 512)


    signal_duration_in_sec = 0.5 # Making 2000 samples per window for raw data sampling rate: 4000
    overlap_ratio = 0.75
    downsampling_ratio = math.ceil(signal_duration_in_sec * SMSMotorData.raw_data_sampling_rate_Hz / (SMSMotorData.cnn_1dinput_len * 2))
    # Making downsampling_ratio 2 for raw data sampling rate 4KHz and 0.5 sec duration

    return smsmotordata_get_datasets(data, load_train, load_test, anomaly_label=False,
                                 downsampling_ratio=downsampling_ratio,
                                 signal_duration_in_sec=signal_duration_in_sec,
                                 overlap_ratio=overlap_ratio,
                                 dc_removed=dc_removed,
                                 dbu_ind=dbu_ind)


def smsmotordata_dur_0_5_overlap_0_75_get_datasets_for_eval(data, load_train=True, load_test=True, dc_removed=True, dbu_ind=False):
    """
    Returns datasets for evaluation purposes: labels are anomaly and/or normal cases
    Note: In this version, test set also includes all available anormal samples
          along with the part of the training set with normal samples
    """
    # NOTE: Signal duration can be modified.
    #       But downsampling ratio is arranged such that
    #       Each signal window contains exactly 1000 (or 1024) samples
    #       (As after FFT, half of this window will be fed to CNN & CNN input size should be 512)


    signal_duration_in_sec = 0.5 # Making 2000 samples per window for raw data sampling rate: 4000
    overlap_ratio = 0.75
    downsampling_ratio = math.ceil(signal_duration_in_sec * SMSMotorData.raw_data_sampling_rate_Hz / (SMSMotorData.cnn_1dinput_len * 2))
    # Making downsampling_ratio 2 for raw data sampling rate 4KHz and 0.5 sec duration

    return smsmotordata_get_datasets(data, load_train, load_test, anomaly_label=True,
                                 downsampling_ratio=downsampling_ratio,
                                 signal_duration_in_sec=signal_duration_in_sec,
                                 overlap_ratio=overlap_ratio,
                                 dc_removed=dc_removed,
                                 dbu_ind=dbu_ind)



def smsmotordata_dur_0_5_overlap_0_75_dbu_ind_get_datasets_for_train(data, load_train=True, load_test=True, dc_removed=True, dbu_ind=True):
    """
    Returns datasets for training purposes: labels are equivalent to signals
    Note: For training, only normal samples are used, test set here is therefore
          part of original set of normal samples (ratio: SMSMotorData.train_ratio).
    """
    # NOTE: Signal duration can be modified.
    #       But downsampling ratio is arranged such that
    #       Each signal window contains exactly 1000 (or 1024) samples
    #       (As after FFT, half of this window will be fed to CNN & CNN input size should be 512)


    signal_duration_in_sec = 0.5 # Making 2000 samples per window for raw data sampling rate: 4000
    overlap_ratio = 0.75
    downsampling_ratio = math.ceil(signal_duration_in_sec * SMSMotorData.raw_data_sampling_rate_Hz / (SMSMotorData.cnn_1dinput_len * 2))
    # Making downsampling_ratio 2 for raw data sampling rate 4KHz and 0.5 sec duration

    return smsmotordata_get_datasets(data, load_train, load_test, anomaly_label=False,
                                 downsampling_ratio=downsampling_ratio,
                                 signal_duration_in_sec=signal_duration_in_sec,
                                 overlap_ratio=overlap_ratio,
                                 dc_removed=dc_removed,
                                 dbu_ind=True)


def smsmotordata_dur_0_5_overlap_0_75_dbu_ind_get_datasets_for_eval(data, load_train=True, load_test=True, dc_removed=True, dbu_ind=True):
    """
    Returns datasets for evaluation purposes: labels are anomaly and/or normal cases
    Note: In this version, test set also includes all available anormal samples
          along with the part of the training set with normal samples
    """
    # NOTE: Signal duration can be modified.
    #       But downsampling ratio is arranged such that
    #       Each signal window contains exactly 1000 (or 1024) samples
    #       (As after FFT, half of this window will be fed to CNN & CNN input size should be 512)


    signal_duration_in_sec = 0.5 # Making 2000 samples per window for raw data sampling rate: 4000
    overlap_ratio = 0.75
    downsampling_ratio = math.ceil(signal_duration_in_sec * SMSMotorData.raw_data_sampling_rate_Hz / (SMSMotorData.cnn_1dinput_len * 2))
    # Making downsampling_ratio 2 for raw data sampling rate 4KHz and 0.5 sec duration

    return smsmotordata_get_datasets(data, load_train, load_test, anomaly_label=True,
                                 downsampling_ratio=downsampling_ratio,
                                 signal_duration_in_sec=signal_duration_in_sec,
                                 overlap_ratio=overlap_ratio,
                                 dc_removed=dc_removed,
                                 dbu_ind=True)

def smsmotordata_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_get_datasets_for_train(data, load_train=True, load_test=True, dc_removed=True):
    """
    Returns datasets for training purposes: labels are equivalent to signals
    Note: For training, only normal samples are used, test set here is therefore
          part of original set of normal samples (ratio: SMSMotorData.train_ratio).
    """
    # NOTE: Signal duration can be modified.
    #       But downsampling ratio is arranged such that
    #       Each signal window contains exactly 1000 (or 1024) samples
    #       (As after FFT, half of this window will be fed to CNN & CNN input size should be 512)


    signal_duration_in_sec = 0.5 # Making 2000 samples per window for raw data sampling rate: 4000
    overlap_ratio = 0.75
    downsampling_ratio = math.ceil(signal_duration_in_sec * SMSMotorData.raw_data_sampling_rate_Hz / (SMSMotorData.cnn_1dinput_len * 2))
    # Making downsampling_ratio 2 for raw data sampling rate 4KHz and 0.5 sec duration

    return smsmotordata_get_datasets(data, load_train, load_test, anomaly_label=False,
                                 downsampling_ratio=downsampling_ratio,
                                 signal_duration_in_sec=signal_duration_in_sec,
                                 overlap_ratio=overlap_ratio,
                                 dc_removed=dc_removed,
                                 dbu_ind=True,
                                 ds_after_fft=True)


def smsmotordata_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_get_datasets_for_eval(data, load_train=True, load_test=True, dc_removed=True):
    """
    Returns datasets for evaluation purposes: labels are anomaly and/or normal cases
    Note: In this version, test set also includes all available anormal samples
          along with the part of the training set with normal samples
    """
    # NOTE: Signal duration can be modified.
    #       But downsampling ratio is arranged such that
    #       Each signal window contains exactly 1000 (or 1024) samples
    #       (As after FFT, half of this window will be fed to CNN & CNN input size should be 512)


    signal_duration_in_sec = 0.5 # Making 2000 samples per window for raw data sampling rate: 4000
    overlap_ratio = 0.75
    downsampling_ratio = math.ceil(signal_duration_in_sec * SMSMotorData.raw_data_sampling_rate_Hz / (SMSMotorData.cnn_1dinput_len * 2))
    # Making downsampling_ratio 2 for raw data sampling rate 4KHz and 0.5 sec duration

    return smsmotordata_get_datasets(data, load_train, load_test, anomaly_label=True,
                                 downsampling_ratio=downsampling_ratio,
                                 signal_duration_in_sec=signal_duration_in_sec,
                                 overlap_ratio=overlap_ratio,
                                 dc_removed=dc_removed,
                                 dbu_ind=True,
                                 ds_after_fft=True)


def smsmotordata_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_withdc_get_datasets_for_train(data, load_train=True, load_test=True):
    """
    Returns datasets for training purposes: labels are equivalent to signals
    Note: For training, only normal samples are used, test set here is therefore
          part of original set of normal samples (ratio: SMSMotorData.train_ratio).
    """
    # NOTE: Signal duration can be modified.
    #       But downsampling ratio is arranged such that
    #       Each signal window contains exactly 1000 (or 1024) samples
    #       (As after FFT, half of this window will be fed to CNN & CNN input size should be 512)


    signal_duration_in_sec = 0.5 # Making 2000 samples per window for raw data sampling rate: 4000
    overlap_ratio = 0.75
    downsampling_ratio = math.ceil(signal_duration_in_sec * SMSMotorData.raw_data_sampling_rate_Hz / (SMSMotorData.cnn_1dinput_len * 2))
    # Making downsampling_ratio 2 for raw data sampling rate 4KHz and 0.5 sec duration

    return smsmotordata_get_datasets(data, load_train, load_test, anomaly_label=False,
                                 downsampling_ratio=downsampling_ratio,
                                 signal_duration_in_sec=signal_duration_in_sec,
                                 overlap_ratio=overlap_ratio,
                                 dc_removed=False,
                                 dbu_ind=True,
                                 ds_after_fft=True)


def smsmotordata_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_withdc_get_datasets_for_eval(data, load_train=True, load_test=True):
    """
    Returns datasets for evaluation purposes: labels are anomaly and/or normal cases
    Note: In this version, test set also includes all available anormal samples
          along with the part of the training set with normal samples
    """
    # NOTE: Signal duration can be modified.
    #       But downsampling ratio is arranged such that
    #       Each signal window contains exactly 1000 (or 1024) samples
    #       (As after FFT, half of this window will be fed to CNN & CNN input size should be 512)


    signal_duration_in_sec = 0.5 # Making 2000 samples per window for raw data sampling rate: 4000
    overlap_ratio = 0.75
    downsampling_ratio = math.ceil(signal_duration_in_sec * SMSMotorData.raw_data_sampling_rate_Hz / (SMSMotorData.cnn_1dinput_len * 2))
    # Making downsampling_ratio 2 for raw data sampling rate 4KHz and 0.5 sec duration

    return smsmotordata_get_datasets(data, load_train, load_test, anomaly_label=True,
                                 downsampling_ratio=downsampling_ratio,
                                 signal_duration_in_sec=signal_duration_in_sec,
                                 overlap_ratio=overlap_ratio,
                                 dc_removed=False,
                                 dbu_ind=True,
                                 ds_after_fft=True)


def smsmotordata_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_withdc_norm_per_ch_get_datasets_for_train(data, load_train=True, load_test=True):
    """
    Returns datasets for training purposes: labels are equivalent to signals
    Note: For training, only normal samples are used, test set here is therefore
          part of original set of normal samples (ratio: SMSMotorData.train_ratio).
    """
    # NOTE: Signal duration can be modified.
    #       But downsampling ratio is arranged such that
    #       Each signal window contains exactly 1000 (or 1024) samples
    #       (As after FFT, half of this window will be fed to CNN & CNN input size should be 512)


    signal_duration_in_sec = 0.5 # Making 2000 samples per window for raw data sampling rate: 4000
    overlap_ratio = 0.75
    downsampling_ratio = math.ceil(signal_duration_in_sec * SMSMotorData.raw_data_sampling_rate_Hz / (SMSMotorData.cnn_1dinput_len * 2))
    # Making downsampling_ratio 2 for raw data sampling rate 4KHz and 0.5 sec duration

    return smsmotordata_get_datasets(data, load_train, load_test, anomaly_label=False,
                                 downsampling_ratio=downsampling_ratio,
                                 signal_duration_in_sec=signal_duration_in_sec,
                                 overlap_ratio=overlap_ratio,
                                 dc_removed=False,
                                 dbu_ind=True,
                                 ds_after_fft=True,
                                 norm_per_ch=True)


def smsmotordata_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_withdc_norm_per_ch_get_datasets_for_eval(data, load_train=True, load_test=True):
    """
    Returns datasets for evaluation purposes: labels are anomaly and/or normal cases
    Note: In this version, test set also includes all available anormal samples
          along with the part of the training set with normal samples
    """
    # NOTE: Signal duration can be modified.
    #       But downsampling ratio is arranged such that
    #       Each signal window contains exactly 1000 (or 1024) samples
    #       (As after FFT, half of this window will be fed to CNN & CNN input size should be 512)


    signal_duration_in_sec = 0.5 # Making 2000 samples per window for raw data sampling rate: 4000
    overlap_ratio = 0.75
    downsampling_ratio = math.ceil(signal_duration_in_sec * SMSMotorData.raw_data_sampling_rate_Hz / (SMSMotorData.cnn_1dinput_len * 2))
    # Making downsampling_ratio 2 for raw data sampling rate 4KHz and 0.5 sec duration

    return smsmotordata_get_datasets(data, load_train, load_test, anomaly_label=True,
                                 downsampling_ratio=downsampling_ratio,
                                 signal_duration_in_sec=signal_duration_in_sec,
                                 overlap_ratio=overlap_ratio,
                                 dc_removed=False,
                                 dbu_ind=True,
                                 ds_after_fft=True,
                                 norm_per_ch=True)


datasets = [
    {
        'name': 'SMSMotorData_dur_0_5_overlap_0_75_ForTrain',
        'input': (3, 512),
        'output': ('signal'),
        'regression': True,
        'loader': smsmotordata_dur_0_5_overlap_0_75_get_datasets_for_train,
    },
    {
        'name': 'SMSMotorData_dur_0_5_overlap_0_75_ForEval',
        'input': (3, 512),
        'output': ('normal', 'anomaly'),
        'loader': smsmotordata_dur_0_5_overlap_0_75_get_datasets_for_eval,
    },
    {
        'name': 'SMSMotorData_dur_0_5_overlap_0_75_dbu_ind_ForTrain',
        'input': (3, 512),
        'output': ('signal'),
        'regression': True,
        'loader': smsmotordata_dur_0_5_overlap_0_75_dbu_ind_get_datasets_for_train,
    },
    {
        'name': 'SMSMotorData_dur_0_5_overlap_0_75_dbu_ind_ForEval',
        'input': (3, 512),
        'output': ('normal', 'anomaly'),
        'loader': smsmotordata_dur_0_5_overlap_0_75_dbu_ind_get_datasets_for_eval,
    },
    {
        'name': 'SMSMotorData_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_ForTrain',
        'input': (3, 512),
        'output': ('signal'),
        'regression': True,
        'loader': smsmotordata_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_get_datasets_for_train,
    },
    {
        'name': 'SMSMotorData_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_ForEval',
        'input': (3, 512),
        'output': ('normal', 'anomaly'),
        'loader': smsmotordata_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_get_datasets_for_eval,
    },
    {
        'name': 'SMSMotorData_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_withdc_ForTrain',
        'input': (3, 512),
        'output': ('signal'),
        'regression': True,
        'loader': smsmotordata_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_withdc_get_datasets_for_train,
    },
    {
        'name': 'SMSMotorData_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_withdc_ForEval',
        'input': (3, 512),
        'output': ('normal', 'anomaly'),
        'loader': smsmotordata_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_withdc_get_datasets_for_eval,
    },
        {
        'name': 'SMSMotorData_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_withdc_normPerCh_ForTrain',
        'input': (3, 512),
        'output': ('signal'),
        'regression': True,
        'loader': smsmotordata_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_withdc_norm_per_ch_get_datasets_for_train,
    },
    {
        'name': 'SMSMotorData_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_withdc_normPerCh_ForEval',
        'input': (3, 512),
        'output': ('normal', 'anomaly'),
        'loader': smsmotordata_dur_0_5_overlap_0_75_dbu_ind_ds_after_fft_withdc_norm_per_ch_get_datasets_for_eval,
    },

    
]
