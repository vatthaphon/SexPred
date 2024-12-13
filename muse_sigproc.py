import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time

outlier_factor_g = 4 # Outlier detection using the unbiased standard deviation.
T_g = 1 # second
fs_g = 256 # The sampling rate of MUSE.

def getCleanEEG(fn_p):

    with open(fn_p, "rb") as fp:
        input_l = pickle.load(fp)

        fp.close()

    TP9_l    = input_l[:, 0, 0]
    AF7_l    = input_l[:, 1, 0]
    AF8_l    = input_l[:, 2, 0]
    TP10_l   = input_l[:, 3, 0]
    AUX_l    = input_l[:, 4, 0]

    TP9_mean_l = np.average(TP9_l)
    AF7_mean_l = np.average(AF7_l)
    AF8_mean_l = np.average(AF8_l)
    TP10_mean_l = np.average(TP10_l)

    assert (not np.isnan(TP9_mean_l)) and (not np.isnan(AF7_mean_l)) and (not np.isnan(AF8_mean_l)) and (not np.isnan(TP10_mean_l))

    TP9_sd_l = np.std(TP9_l, ddof=1)
    AF7_sd_l = np.std(AF7_l, ddof=1)
    AF8_sd_l = np.std(AF8_l, ddof=1)
    TP10_sd_l = np.std(TP10_l, ddof=1)

    TP9_l = TP9_l - TP9_mean_l
    AF7_l = AF7_l - AF7_mean_l
    AF8_l = AF8_l - AF8_mean_l
    TP10_l = TP10_l - TP10_mean_l

    n_samples = len(TP9_l)
    n_secs = int(np.floor(n_samples / (fs_g*T_g)))

    reshape = lambda a, n_secs, fs_g: np.transpose(np.reshape(a, (n_secs, fs_g))) # [fs_g, n_secs]

    TP9_epochs_l = reshape(TP9_l, n_secs, (fs_g*T_g)) 
    AF7_epochs_l = reshape(AF7_l, n_secs, (fs_g*T_g))
    AF8_epochs_l = reshape(AF8_l, n_secs, (fs_g*T_g))
    TP10_epochs_l = reshape(TP10_l, n_secs, (fs_g*T_g))

    # print("|TP9| <= ", outlier_factor_g*TP9_sd_l)
    is_outliers_TP9_l = np.abs(TP9_epochs_l) > (outlier_factor_g*TP9_sd_l)
    is_outliers_TP9_l = np.sum(is_outliers_TP9_l, axis=0) > 0

    # print("|AF7| <= ", outlier_factor_g*AF7_sd_l)
    is_outliers_AF7_l = np.abs(AF7_epochs_l) > (outlier_factor_g*AF7_sd_l)
    is_outliers_AF7_l = np.sum(is_outliers_AF7_l, axis=0) > 0

    # print("|AF8| <= ", outlier_factor_g*AF8_sd_l)
    is_outliers_AF8_l = np.abs(AF8_epochs_l) > (outlier_factor_g*AF8_sd_l)
    is_outliers_AF8_l = np.sum(is_outliers_AF8_l, axis=0) > 0

    # print("|TP10| <= ", outlier_factor_g*TP10_sd_l)
    is_outliers_TP10_l = np.abs(TP10_epochs_l) > (outlier_factor_g*TP10_sd_l)
    is_outliers_TP10_l = np.sum(is_outliers_TP10_l, axis=0) > 0

    is_outliers_epoch = is_outliers_TP9_l | is_outliers_AF7_l | is_outliers_AF8_l | is_outliers_TP10_l

    TP9_clean_epochs_l = TP9_epochs_l[:, ~is_outliers_epoch]
    AF7_clean_epochs_l = AF7_epochs_l[:, ~is_outliers_epoch]
    AF8_clean_epochs_l = AF8_epochs_l[:, ~is_outliers_epoch]
    TP10_clean_epochs_l = TP10_epochs_l[:, ~is_outliers_epoch]
    
    assert np.max(np.abs(TP9_clean_epochs_l)) <=  outlier_factor_g*TP9_sd_l
    assert np.max(np.abs(AF7_clean_epochs_l)) <=  outlier_factor_g*AF7_sd_l
    assert np.max(np.abs(AF8_clean_epochs_l)) <=  outlier_factor_g*AF8_sd_l
    assert np.max(np.abs(TP10_clean_epochs_l)) <=  outlier_factor_g*TP10_sd_l

    # print("Max clean |TP9|: ", np.max(np.abs(TP9_clean_epochs_l)))
    # print("Max clean |AF7|: ", np.max(np.abs(AF7_clean_epochs_l)))
    # print("Max clean |AF8|: ", np.max(np.abs(AF8_clean_epochs_l)))
    # print("Max clean |TP10|: ", np.max(np.abs(TP10_clean_epochs_l)))

    flatten = lambda a : np.ravel(np.transpose(a))
    rescale = lambda a : (((a - np.min(a)) / (np.max(a) - np.min(a))) - 0.5)*2 # Rescale the value to [-1, 1]

    tmp_TP9_l = rescale(flatten(TP9_clean_epochs_l))
    tmp_AF7_l = rescale(flatten(AF7_clean_epochs_l))
    tmp_AF8_l = rescale(flatten(AF8_clean_epochs_l))
    tmp_TP10_l = rescale(flatten(TP10_clean_epochs_l))

    n_samples_clean = len(tmp_TP9_l)
    n_secs_clean = int(np.floor(n_samples_clean / (fs_g*T_g)))

    TP9_rescaled_l = reshape(tmp_TP9_l, n_secs_clean, (fs_g*T_g)) 
    AF7_rescaled_l = reshape(tmp_AF7_l, n_secs_clean, (fs_g*T_g))
    AF8_rescaled_l = reshape(tmp_AF8_l, n_secs_clean, (fs_g*T_g))
    TP10_rescaled_l = reshape(tmp_TP10_l, n_secs_clean, (fs_g*T_g))


    tmp_TP9_l = flatten(TP9_rescaled_l)
    tmp_AF7_l = flatten(AF7_rescaled_l)
    tmp_AF8_l = flatten(AF8_rescaled_l)
    tmp_TP10_l = flatten(TP10_rescaled_l)

    print("#Epoch: ", np.shape(TP9_epochs_l), "->", np.shape(TP9_clean_epochs_l))

    EEG = np.stack((TP9_rescaled_l, AF7_rescaled_l, AF8_rescaled_l, TP10_rescaled_l), axis=2)

    # print("EEG: ", np.shape(EEG))

    return EEG


def prepeeg4ML():

    ## Data from MW projects.
    dat_dir_l = 

    dir_list_l = os.listdir(dat_dir_l)    

    for dir in dir_list_l:
        beg_eeg_l = os.path.join(dat_dir_l, dir, dir + "_Before_EEG_data")
        aft_eeg_l = os.path.join(dat_dir_l, dir, dir + "_After_EEG_data")

        EEG = None

        if os.path.isfile(beg_eeg_l):
            print(beg_eeg_l)
            EEG = getCleanEEG(beg_eeg_l)

        if os.path.isfile(aft_eeg_l):
            print(aft_eeg_l)
            EEG_aft_l = getCleanEEG(aft_eeg_l)

            if EEG is None:
                EEG = EEG_aft_l
            else:
                EEG = np.concatenate((EEG, EEG_aft_l), axis=1)

        if EEG is not None:
            print("EEG: ", np.shape(EEG))

            directory = os.path.join(dat_dir_l, dir, "SexPred")
            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(os.path.join(directory, "EEG_" + str(T_g) + "sec_epochs") , "wb+") as fp:
                pickle.dump(EEG, fp)
                fp.close()

    ## Data from depression projects.
    dat_dir_l = 

    dir_list_l = os.listdir(dat_dir_l)    

    for dir in dir_list_l:

        eeg_l = os.path.join(dat_dir_l, dir, dir + "_EEG.dat")

        if os.path.isfile(eeg_l):
            print(eeg_l)

            try:
                EEG = getCleanEEG(eeg_l)

                # print("EEG: ", np.shape(EEG))

                directory = os.path.join(dat_dir_l, dir, "SexPred")
                if not os.path.exists(directory):
                    os.makedirs(directory)

                fn_l = os.path.join(directory, "EEG_" + str(T_g) + "sec_epochs")
                with open(fn_l , "wb+") as fp:
                    pickle.dump(EEG, fp)
                    fp.close() 
            except:
                print(dir, " is corrupted.")

def test_read_eeg():

    fn_l = 
    
    with open(fn_l, "rb") as fp:
        input_l = pickle.load(fp)

        fp.close()

    TP9_l    = input_l[:, 0, 0]
    AF7_l    = input_l[:, 1, 0]
    AF8_l    = input_l[:, 2, 0]
    TP10_l   = input_l[:, 3, 0]
    AUX_l    = input_l[:, 4, 0]

    TP9_mean_l = np.average(TP9_l)
    AF7_mean_l = np.average(AF7_l)
    AF8_mean_l = np.average(AF8_l)
    TP10_mean_l = np.average(TP10_l)

    assert (not np.isnan(TP9_mean_l)) and (not np.isnan(AF7_mean_l)) and (not np.isnan(AF8_mean_l)) and (not np.isnan(TP10_mean_l))

    TP9_sd_l = np.std(TP9_l, ddof=1)
    AF7_sd_l = np.std(AF7_l, ddof=1)
    AF8_sd_l = np.std(AF8_l, ddof=1)
    TP10_sd_l = np.std(TP10_l, ddof=1)

    TP9_l = TP9_l - TP9_mean_l
    AF7_l = AF7_l - AF7_mean_l
    AF8_l = AF8_l - AF8_mean_l
    TP10_l = TP10_l - TP10_mean_l

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    fig.suptitle("EEG")
    ax1.plot(TP9_l)
    ax2.plot(AF7_l)        
    ax3.plot(AF8_l)        
    ax4.plot(TP10_l)   

    n_samples = len(TP9_l)
    n_secs = int(np.floor(n_samples / fs_g))

    reshape = lambda a, n_secs, fs_g: np.transpose(np.reshape(a, (n_secs, fs_g))) # [fs_g, n_secs]

    TP9_epochs_l = reshape(TP9_l, n_secs, fs_g) 
    AF7_epochs_l = reshape(AF7_l, n_secs, fs_g)
    AF8_epochs_l = reshape(AF8_l, n_secs, fs_g)
    TP10_epochs_l = reshape(TP10_l, n_secs, fs_g)

    print("|TP9| <= ", outlier_factor_g*TP9_sd_l)
    is_outliers_TP9_l = np.abs(TP9_epochs_l) > (outlier_factor_g*TP9_sd_l)
    is_outliers_TP9_l = np.sum(is_outliers_TP9_l, axis=0) > 0

    print("|AF7| <= ", outlier_factor_g*AF7_sd_l)
    is_outliers_AF7_l = np.abs(AF7_epochs_l) > (outlier_factor_g*AF7_sd_l)
    is_outliers_AF7_l = np.sum(is_outliers_AF7_l, axis=0) > 0

    print("|AF8| <= ", outlier_factor_g*AF8_sd_l)
    is_outliers_AF8_l = np.abs(AF8_epochs_l) > (outlier_factor_g*AF8_sd_l)
    is_outliers_AF8_l = np.sum(is_outliers_AF8_l, axis=0) > 0

    print("|TP10| <= ", outlier_factor_g*TP10_sd_l)
    is_outliers_TP10_l = np.abs(TP10_epochs_l) > (outlier_factor_g*TP10_sd_l)
    is_outliers_TP10_l = np.sum(is_outliers_TP10_l, axis=0) > 0

    is_outliers_epoch = is_outliers_TP9_l | is_outliers_AF7_l | is_outliers_AF8_l | is_outliers_TP10_l

    TP9_clean_epochs_l = TP9_epochs_l[:, ~is_outliers_epoch]
    AF7_clean_epochs_l = AF7_epochs_l[:, ~is_outliers_epoch]
    AF8_clean_epochs_l = AF8_epochs_l[:, ~is_outliers_epoch]
    TP10_clean_epochs_l = TP10_epochs_l[:, ~is_outliers_epoch]
    
    assert np.max(np.abs(TP9_clean_epochs_l)) <=  outlier_factor_g*TP9_sd_l
    assert np.max(np.abs(AF7_clean_epochs_l)) <=  outlier_factor_g*AF7_sd_l
    assert np.max(np.abs(AF8_clean_epochs_l)) <=  outlier_factor_g*AF8_sd_l
    assert np.max(np.abs(TP10_clean_epochs_l)) <=  outlier_factor_g*TP10_sd_l

    print("Max clean |TP9|: ", np.max(np.abs(TP9_clean_epochs_l)))
    print("Max clean |AF7|: ", np.max(np.abs(AF7_clean_epochs_l)))
    print("Max clean |AF8|: ", np.max(np.abs(AF8_clean_epochs_l)))
    print("Max clean |TP10|: ", np.max(np.abs(TP10_clean_epochs_l)))

    flatten = lambda a : np.ravel(np.transpose(a))
    rescale = lambda a : (((a - np.min(a)) / (np.max(a) - np.min(a))) - 0.5)*2 # Rescale the value to [-1, 1]

    tmp_TP9_l = rescale(flatten(TP9_clean_epochs_l))
    tmp_AF7_l = rescale(flatten(AF7_clean_epochs_l))
    tmp_AF8_l = rescale(flatten(AF8_clean_epochs_l))
    tmp_TP10_l = rescale(flatten(TP10_clean_epochs_l))

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    # fig.suptitle("EEG")
    # ax1.plot(tmp_TP9_l)
    # ax2.plot(tmp_AF7_l)        
    # ax3.plot(tmp_AF8_l)        
    # ax4.plot(tmp_TP10_l)       

    n_samples_clean = len(tmp_TP9_l)
    n_secs_clean = int(np.floor(n_samples_clean / fs_g))

    TP9_rescaled_l = reshape(tmp_TP9_l, n_secs_clean, fs_g) 
    AF7_rescaled_l = reshape(tmp_AF7_l, n_secs_clean, fs_g)
    AF8_rescaled_l = reshape(tmp_AF8_l, n_secs_clean, fs_g)
    TP10_rescaled_l = reshape(tmp_TP10_l, n_secs_clean, fs_g)


    tmp_TP9_l = flatten(TP9_rescaled_l)
    tmp_AF7_l = flatten(AF7_rescaled_l)
    tmp_AF8_l = flatten(AF8_rescaled_l)
    tmp_TP10_l = flatten(TP10_rescaled_l)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    fig.suptitle("EEG")
    ax1.plot(tmp_TP9_l)
    ax2.plot(tmp_AF7_l)        
    ax3.plot(tmp_AF8_l)        
    ax4.plot(tmp_TP10_l)       

    print("#Epoch: ", np.shape(TP9_epochs_l), "->", np.shape(TP9_clean_epochs_l))

    EEG = np.stack((TP9_rescaled_l, AF7_rescaled_l, AF8_rescaled_l, TP10_rescaled_l), axis=2)

    print("EEG: ", np.shape(EEG))


if __name__ == "__main__":
    start_time_l = time.ctime()

    # test_read_eeg()

    prepeeg4ML()


    print(start_time_l, "-- Start")
    print(time.ctime(), "-- Stop")

    plt.show()