import logging
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import os
import optuna
import optuna.visualization as vis
import pandas as pd
import pickle
import random
import shutil
import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import time

from mlflow.models.signature import infer_signature
from numbers_parser import Document
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

import muse_models as av_ml

########## Default parameters ##########
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

# fileHandler = logging.FileHandler("{0}/{1}.log".format(os.getcwd(), "ML"))
# fileHandler.setFormatter(logFormatter)
# rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

rootLogger.setLevel(logging.INFO)

is_replace_run_g = False # By default, we do not delete the old runs.

female_code_g = 0
male_code_g = 1

is_test_run_n_epochs = False

########## Tweaked parameters ##########
is_replace_run_g = True

is_test_run_n_epochs = True

########## MLFlow ##########
def link_mlflow_local():
    rootLogger.info("Set tracking url")
    mlflow.set_tracking_uri("http://localhost:5000")

def link_mlflow_databricks():

    rootLogger.info("Logging Databricks Community Edition")
    mlflow.login()

    rootLogger.info("Set tracking url")
    mlflow.set_tracking_uri("databricks")

def set_mlflow_exp(exp_name):

    rootLogger.info("Set experiment to " +  exp_name)
    mlflow.set_experiment(exp_name)

def set_mlflow_run(exp_name, run_name):

    if is_replace_run_g:
        del_run_id_by_name(exp_name, run_name)        

    rootLogger.info("Set run name to " +  run_name)
    mlflow.set_tag("mlflow.runName", run_name)  

def del_run_id_by_name(exp_name, run_name):
    
    run_ids = get_run_id_by_name(exp_name, run_name)

    if run_ids is not None:

        # print(run_ids)

        for run_id in run_ids:

            # print(run_id)
            mlflow.delete_run(run_id)
            rootLogger.info("Delete run success.")
    else:
        rootLogger.info("Not found run \"" + run_name + "\"")

def get_run_id_by_name(exp_name, run_name):

    exps = mlflow.search_experiments()
    # print(mlflow.search_experiments())

    exp_id = None
    for exp in exps:
        if exp.name == ("/" + exp_name):

            exp_id = exp.experiment_id

            break

    if exp_id is not None:

        try:

            run_pds = mlflow.search_runs(experiment_ids=[exp_id])        

            # print(run_pds.columns.tolist())
            # print(run_pds.loc[:,"tags.mlflow.runName"])

            matching_rows = run_pds[run_pds["tags.mlflow.runName"] == run_name]

            matching_run_ids = matching_rows["run_id"]

            return matching_run_ids.tolist()

        except:

            return None

    else:

        rootLogger.info(f"Not found {exp_name}.")


    return None

########## Hyperparameter optimization ##########
def objective(trial, exp_name_p):

    # Access the trial ID
    trial_id_l = trial.number
    rootLogger.info(f"Trial ID: {trial_id_l}")

    sex_dict_l = loadSex()
    EEG_dict_l = loadEEGs()

    sorted_names_l = sorted(EEG_dict_l.keys())

    nb_classes_l = 2 # The number of classs to be predicted.
    batch_size_l = 32
    epochs_l = 100
    # epochs_l = 3
    lr_l = 0.001

    valid_percentage_l = 20 # The percentage of subjects to be used as a validation set.
    test_percentage_l = 20 # The percentage of subjects to be used as a test set.

    n_test_l = int(np.floor(len(sorted_names_l)*test_percentage_l / 100))
    n_valid_l = int(np.floor(len(sorted_names_l)*valid_percentage_l / 100))
    n_train_l = len(sorted_names_l) - (n_valid_l + n_test_l)

    rootLogger.info(f"#Train: {n_train_l}, #Valid: {n_valid_l}, #Test: {n_test_l}")

    np.random.seed(0)
    random.seed(0)

    test_names_l = random.sample(sorted_names_l, n_test_l)
    train_valid_names_l = sorted([subj for subj in sorted_names_l if subj not in test_names_l])
    valid_names_l = random.sample(train_valid_names_l, n_valid_l)
    train_names_l = sorted([subj for subj in train_valid_names_l if subj not in valid_names_l])

    x_trains_l, y_trains_l, x_valids_l, y_valids_l = gen_train_valid(EEG_dict_l, sex_dict_l, train_names_l, valid_names_l)

    Y_train_l   = to_categorical(y_trains_l, nb_classes_l)               
    Y_val_l     = to_categorical(y_valids_l, nb_classes_l)               

    x_test_l, y_test_l = gen_test(EEG_dict_l, sex_dict_l, test_names_l)
    Y_test_l = to_categorical(y_test_l, nb_classes_l)   

    model_params_p = {}
    model_params_p["optuna_trial"] = trial
    model_params_p["mlflow"] = mlflow

    run_name_l = "hype_opt_" + str(trial_id_l)
    del_run_id_by_name(exp_name_p, run_name_l)

    mlflow.start_run(run_name=run_name_l)

    model_l, desc_l, _ = loadModel(model_params_p)

    rootLogger.info(desc_l)

    av_ml.AV_Model_compile(model_p=model_l, optimizer_p=keras.optimizers.Adam(lr_l), loss_p="categorical_crossentropy", metrics_p=["accuracy"])


    # with mlflow.start_run(run_name=run_name_l) as new_run:    

    history_l = av_ml.AV_Model_train_allbatches(    model_p=model_l,
                                                    x_p=x_trains_l,                 # Input data. It could be:
                                                                                    # A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
                                                                                    # A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
                                                                                    # A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample weights).
                                                                                    # None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
                                                    y_p=Y_train_l,                  # Target data. Like the input data x, it could be either Numpy array(s), framework-native tensor(s), list of Numpy arrays (if the model has multiple outputs) or None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays. If x is a generator, or keras.utils.Sequence instance, y should not be specified (since targets will be obtained from x).
                                                    batch_size_p=batch_size_l,      # Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of symbolic tensors, generators, or Sequence instances (since they generate batches).
                                                    epochs_p=epochs_l,              # Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
                                                    verbose_p=1,                    # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 
                                                    callbacks_p=[MLFlowCallback()],               # List of keras.callbacks.Callback instances. List of callbacks to apply during training and validation (if ). See callbacks.
                                                    validation_split_p=0.0,         # Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling. This argument is not supported when x is a generator or Sequence instance.
                                                    validation_data_p=(x_valids_l, Y_val_l),         # Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. validation_data will override validation_split. validation_data could be: - tuple (x_val, y_val) of Numpy arrays or tensors - tuple (x_val, y_val, val_sample_weights) of Numpy arrays - dataset or a dataset iterator
                                                                                    # For the first two cases, batch_size must be provided. For the last case, validation_steps must be provided.
                                                    shuffle_p=True,                 # Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
                                                    class_weight_p=None,            # Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
                                                    sample_weight_p=None,           # Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile(). This argument is not supported when x generator, or Sequence instance, instead provide the sample_weights as the third element of x.
                                                    initial_epoch_p=0,              # Integer. Epoch at which to start training (useful for resuming a previous training run).
                                                    steps_per_epoch_p=None,         # Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
                                                    validation_steps_p=None,        # Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.
                                                    validation_freq_p=1,            # Only relevant if validation data is provided. Integer or list/tuple/set. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a list, tuple, or set, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.
                                                    max_queue_size_p=10,            # Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
                                                    workers_p=1,                    # Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
                                                    use_multiprocessing_p=False     # Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.
                                                    )   

    mlflow.log_param("learning_rate", lr_l)
    mlflow.log_param("batch_size", batch_size_l)
    mlflow.log_param("num_epochs", epochs_l)        
    mlflow.log_param("Subjects for validation in percentage", valid_percentage_l)
    mlflow.log_param("Subjects for test in percentage", test_percentage_l)

    score_valid_l = model_l.evaluate(x_valids_l, Y_val_l, verbose=0)
    valid_acc = score_valid_l[1]
    rootLogger.info(f"Validate accuracy: {valid_acc}") 

    score_valid_l = model_l.evaluate(x_test_l, Y_test_l, verbose=0)
    test_acc = score_valid_l[1]
    rootLogger.info(f"Test accuracy: {test_acc}") 

    description_l = f"Validation accuracy: {valid_acc}; Test accuracy: {test_acc}"
    mlflow.set_tag("mlflow.note.content", description_l)
    mlflow.set_tag("Data", "Eye-Closed Resting EEG in the depression and mind-wandering projects.")

    mlflow.end_run()

    return valid_acc

def hype_opt():

    _, _, exp_name_l = loadModel()
    set_mlflow_exp(exp_name_l)

    # n_trials_l = 100
    n_trials_l = 50
    # n_trials_l = 5

    ## Create a study object
    study = optuna.create_study(study_name=exp_name_l, storage="sqlite:///" + exp_name_l + ".db", direction="maximize", load_if_exists=True) # Tree-structured Parzen Estimator    

    if len(study.trials) > 0:

        last_trial_l = study.trials[-1]

        if last_trial_l.state == optuna.trial.TrialState.FAIL: 

            study.enqueue_trial(last_trial_l.params)

            ## Count the number of success trials.
            n_success_trials_l = 0

            for i_trial, trial in enumerate(study.trials):

                if trial.state == optuna.trial.TrialState.COMPLETE: 

                    rootLogger.info(f"Hyp op: Trial {i_trial} is complete.")

                    n_success_trials_l += 1


            n_eff_trials_l = n_trials_l - n_success_trials_l

            rootLogger.info(f"Hyperparameters Optimization: completed {n_success_trials_l} trials of {n_trials_l}")


            if n_eff_trials_l > 0:

                ## Optimize the objective function
                objective_args = lambda trial: objective(trial, exp_name_l)
                study.optimize(objective_args, n_trials=n_eff_trials_l)    


    else:

        ## Optimize the objective function
        objective_args = lambda trial: objective(trial, exp_name_l)
        study.optimize(objective_args, n_trials=n_trials_l)    

    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)    

    ## Save the study object to a file with pickle
    with open(exp_name_l + ".pkl", "wb") as f:
        pickle.dump(study, f)    


def hype_opt_vis():

    _, _, exp_name_l = loadModel()

    with open(exp_name_l + ".pkl", "rb") as fp:
        study_l = pickle.load(fp)    

    vis.plot_optimization_history(study_l).show(renderer="browser")
    vis.plot_param_importances(study_l).show(renderer="browser")
    vis.plot_slice(study_l).show(renderer="browser")


def hype_opt_best_network(isFinalModel=True):

    _, _, exp_name_l = loadModel()
    set_mlflow_exp(exp_name_l)

    sex_dict_l = loadSex()
    EEG_dict_l = loadEEGs()

    sorted_names_l = sorted(EEG_dict_l.keys())

    nb_classes_l = 2 # The number of classs to be predicted.
    batch_size_l = 32
    epochs_l = 100
    # epochs_l = 3
    lr_l = 0.001

    if isFinalModel:

        ## Ingestion
        valid_percentage_l = 20 # The percentage of subjects to be used as a validation set.        
        test_percentage_l = 20 # The percentage of subjects to be used as a test set.

        n_test_l = int(np.floor(len(sorted_names_l)*test_percentage_l / 100))
        n_valid_l = int(np.floor(len(sorted_names_l)*valid_percentage_l / 100))
        n_train_l = len(sorted_names_l) - (n_valid_l + n_test_l)

        np.random.seed(0)
        random.seed(0)
        test_names_l = random.sample(sorted_names_l, n_test_l)
        train_valid_names_l = sorted([subj for subj in sorted_names_l if subj not in test_names_l])        
        valid_names_l = random.sample(train_valid_names_l, n_valid_l)
        train_names_l = sorted([subj for subj in train_valid_names_l if subj not in valid_names_l])


        # rootLogger.info(f"Train on all subjets: {len(train_valid_names_l)}")
        rootLogger.info(f"#Train: {n_train_l}, #Valid: {n_valid_l}, #Test: {n_test_l}")

        # x_trains_l, y_trains_l, _, _ = gen_train_valid(EEG_dict_l, sex_dict_l, train_valid_names_l, None)
        x_trains_l, y_trains_l, _, _ = gen_train_valid(EEG_dict_l, sex_dict_l, train_names_l, valid_names_l)        

        Y_train_l   = to_categorical(y_trains_l, nb_classes_l) 

        x_test_l, y_test_l = gen_test(EEG_dict_l, sex_dict_l, test_names_l)
        Y_test_l = to_categorical(y_test_l, nb_classes_l)   


        ## Model trainning
        best_params = {}
        best_params["reg_l2_l"] = 0.19555368296821243
        best_params["dropout_rate_Conv_l"] = 0.5924693887532844

        model_params_p = {}
        model_params_p["best_params"] = best_params

        run_name_l = "hype_opt_best_network_final"
        del_run_id_by_name(exp_name_l, run_name_l)

        with mlflow.start_run(run_name=run_name_l) as run:

            model_l, desc_l, _ = loadModel(model_params_p)

            rootLogger.info(desc_l)

            av_ml.AV_Model_compile(model_p=model_l, optimizer_p=keras.optimizers.Adam(lr_l), loss_p="categorical_crossentropy", metrics_p=["accuracy"])


            # with mlflow.start_run(run_name=run_name_l) as new_run:    

            history_l = av_ml.AV_Model_train_allbatches(    model_p=model_l,
                                                            x_p=x_trains_l,                 # Input data. It could be:
                                                                                            # A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
                                                                                            # A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
                                                                                            # A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample weights).
                                                                                            # None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
                                                            y_p=Y_train_l,                  # Target data. Like the input data x, it could be either Numpy array(s), framework-native tensor(s), list of Numpy arrays (if the model has multiple outputs) or None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays. If x is a generator, or keras.utils.Sequence instance, y should not be specified (since targets will be obtained from x).
                                                            batch_size_p=batch_size_l,      # Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of symbolic tensors, generators, or Sequence instances (since they generate batches).
                                                            epochs_p=epochs_l,              # Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
                                                            verbose_p=1,                    # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 
                                                            callbacks_p=[MLFlowCallback()],               # List of keras.callbacks.Callback instances. List of callbacks to apply during training and validation (if ). See callbacks.
                                                            # validation_split_p=0.0,         # Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling. This argument is not supported when x is a generator or Sequence instance.
                                                            # validation_data_p=(x_valids_l, Y_val_l),         # Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. validation_data will override validation_split. validation_data could be: - tuple (x_val, y_val) of Numpy arrays or tensors - tuple (x_val, y_val, val_sample_weights) of Numpy arrays - dataset or a dataset iterator
                                                                                            # For the first two cases, batch_size must be provided. For the last case, validation_steps must be provided.
                                                            shuffle_p=True,                 # Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
                                                            class_weight_p=None,            # Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
                                                            sample_weight_p=None,           # Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile(). This argument is not supported when x generator, or Sequence instance, instead provide the sample_weights as the third element of x.
                                                            initial_epoch_p=0,              # Integer. Epoch at which to start training (useful for resuming a previous training run).
                                                            steps_per_epoch_p=None,         # Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
                                                            # validation_steps_p=None,        # Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.
                                                            # validation_freq_p=1,            # Only relevant if validation data is provided. Integer or list/tuple/set. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a list, tuple, or set, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.
                                                            max_queue_size_p=10,            # Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
                                                            workers_p=1,                    # Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
                                                            use_multiprocessing_p=False     # Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.
                                                            )   

            mlflow.log_param("learning_rate", lr_l)
            mlflow.log_param("batch_size", batch_size_l)
            mlflow.log_param("num_epochs", epochs_l)        
            mlflow.log_param("Subjects for validation in percentage", 0)
            mlflow.log_param("Subjects for test in percentage", 100)

            description_l = f"It is trained on all subjects."
            mlflow.set_tag("mlflow.note.content", description_l)
            mlflow.set_tag("Data", "Eye-Closed Resting EEG in the depression and mind-wandering projects.")

            score_test_l = model_l.evaluate(x_test_l, Y_test_l, verbose=0)
            print('Test loss:', score_test_l[0])
            print('Test accuracy:', score_test_l[1]) 

            example_input_l = x_test_l
            example_output_l = model_l.predict(example_input_l)
            signature_l = infer_signature(example_input_l, example_output_l)
            rootLogger.info("Signature of the model:")
            print(signature_l)

            try:
                ## Delete the old model.
                save_model_dir_l = "model"

                if os.path.exists("./" + save_model_dir_l):
                    
                    shutil.rmtree("./" + save_model_dir_l)

                    rootLogger.info("Delete the old model.")

                else:

                    rootLogger.info(f"Couldn't find ./{save_model_dir_l}.")

                ## Save the new model.
                rootLogger.info("Attemp for logging the model.")    
                mlflow.tensorflow.log_model(model=model_l, artifact_path=save_model_dir_l, signature=signature_l) # 1) Store the model in ./mlartifacts 2) Log the model.
                rootLogger.info("Successful for logging the model.")

                ## Register model.
                rootLogger.info("Attemp for registering the model.")

                model_uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(model_uri, 'SexPredictor')

                rootLogger.info("Successful for registering the model.")

                logging.info("MLflow tracking completed successfully")                

            except:

                rootLogger.error("Fail to record the model.")                    

    else:        

        valid_percentage_l = 20 # The percentage of subjects to be used as a validation set.
        test_percentage_l = 20 # The percentage of subjects to be used as a test set.

        n_test_l = int(np.floor(len(sorted_names_l)*test_percentage_l / 100))
        n_valid_l = int(np.floor(len(sorted_names_l)*valid_percentage_l / 100))
        n_train_l = len(sorted_names_l) - (n_valid_l + n_test_l)

        rootLogger.info(f"#Train: {n_train_l}, #Valid: {n_valid_l}, #Test: {n_test_l}")

        np.random.seed(0)
        random.seed(0)

        test_names_l = random.sample(sorted_names_l, n_test_l)
        train_valid_names_l = sorted([subj for subj in sorted_names_l if subj not in test_names_l])
        valid_names_l = random.sample(train_valid_names_l, n_valid_l)
        train_names_l = sorted([subj for subj in train_valid_names_l if subj not in valid_names_l])

        x_trains_l, y_trains_l, x_valids_l, y_valids_l = gen_train_valid(EEG_dict_l, sex_dict_l, train_names_l, valid_names_l)

        Y_train_l   = to_categorical(y_trains_l, nb_classes_l)               
        Y_val_l     = to_categorical(y_valids_l, nb_classes_l)               

        x_test_l, y_test_l = gen_test(EEG_dict_l, sex_dict_l, test_names_l)
        Y_test_l = to_categorical(y_test_l, nb_classes_l)   

        best_params = {}
        best_params["reg_l2_l"] = 0.19555368296821243
        best_params["dropout_rate_Conv_l"] = 0.5924693887532844

        model_params_p = {}
        model_params_p["best_params"] = best_params

        run_name_l = "hype_opt_best_network"
        del_run_id_by_name(exp_name_l, run_name_l)

        mlflow.start_run(run_name=run_name_l)

        model_l, desc_l, _ = loadModel(model_params_p)

        rootLogger.info(desc_l)

        av_ml.AV_Model_compile(model_p=model_l, optimizer_p=keras.optimizers.Adam(lr_l), loss_p="categorical_crossentropy", metrics_p=["accuracy"])


        # with mlflow.start_run(run_name=run_name_l) as new_run:    

        history_l = av_ml.AV_Model_train_allbatches(    model_p=model_l,
                                                        x_p=x_trains_l,                 # Input data. It could be:
                                                                                        # A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
                                                                                        # A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
                                                                                        # A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample weights).
                                                                                        # None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
                                                        y_p=Y_train_l,                  # Target data. Like the input data x, it could be either Numpy array(s), framework-native tensor(s), list of Numpy arrays (if the model has multiple outputs) or None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays. If x is a generator, or keras.utils.Sequence instance, y should not be specified (since targets will be obtained from x).
                                                        batch_size_p=batch_size_l,      # Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of symbolic tensors, generators, or Sequence instances (since they generate batches).
                                                        epochs_p=epochs_l,              # Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
                                                        verbose_p=1,                    # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 
                                                        callbacks_p=[MLFlowCallback()],               # List of keras.callbacks.Callback instances. List of callbacks to apply during training and validation (if ). See callbacks.
                                                        validation_split_p=0.0,         # Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling. This argument is not supported when x is a generator or Sequence instance.
                                                        validation_data_p=(x_valids_l, Y_val_l),         # Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. validation_data will override validation_split. validation_data could be: - tuple (x_val, y_val) of Numpy arrays or tensors - tuple (x_val, y_val, val_sample_weights) of Numpy arrays - dataset or a dataset iterator
                                                                                        # For the first two cases, batch_size must be provided. For the last case, validation_steps must be provided.
                                                        shuffle_p=True,                 # Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
                                                        class_weight_p=None,            # Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
                                                        sample_weight_p=None,           # Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile(). This argument is not supported when x generator, or Sequence instance, instead provide the sample_weights as the third element of x.
                                                        initial_epoch_p=0,              # Integer. Epoch at which to start training (useful for resuming a previous training run).
                                                        steps_per_epoch_p=None,         # Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
                                                        validation_steps_p=None,        # Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.
                                                        validation_freq_p=1,            # Only relevant if validation data is provided. Integer or list/tuple/set. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a list, tuple, or set, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.
                                                        max_queue_size_p=10,            # Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
                                                        workers_p=1,                    # Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
                                                        use_multiprocessing_p=False     # Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.
                                                        )   

        mlflow.log_param("learning_rate", lr_l)
        mlflow.log_param("batch_size", batch_size_l)
        mlflow.log_param("num_epochs", epochs_l)        
        mlflow.log_param("Subjects for validation in percentage", valid_percentage_l)
        mlflow.log_param("Subjects for test in percentage", test_percentage_l)

        score_valid_l = model_l.evaluate(x_valids_l, Y_val_l, verbose=0)
        valid_acc = score_valid_l[1]
        rootLogger.info(f"Validate accuracy: {valid_acc}") 

        score_valid_l = model_l.evaluate(x_test_l, Y_test_l, verbose=0)
        test_acc = score_valid_l[1]
        rootLogger.info(f"Test accuracy: {test_acc}") 

        description_l = f"Validation accuracy: {valid_acc}; Test accuracy: {test_acc}"
        mlflow.set_tag("mlflow.note.content", description_l)
        mlflow.set_tag("Data", "Eye-Closed Resting EEG in the depression and mind-wandering projects.")

        mlflow.end_run()

########## Train ##########
def gen_test(EEG_dict_p, sex_dict_p, test_names_p):

    x_tests_l = None
    y_tests_l = None

    ## Generate the train data
    rootLogger.info("Generatte the test data.")

    for name in test_names_p:

        eeg_l = EEG_dict_p[name]
        sex_l = sex_dict_p[name]        

        n_batch_l = np.shape(eeg_l)[1]
        sex_l = [sex_l] * n_batch_l

        if x_tests_l is None:
            x_tests_l = eeg_l
        else:        
            x_tests_l = np.concatenate((x_tests_l, eeg_l), axis=1)

        if y_tests_l is None:
            y_tests_l = sex_l
        else:
            y_tests_l = y_tests_l + sex_l

    x_tests_l = np.swapaxes(x_tests_l, 0, 1) # Make [batch, sample points, electrodes]

    indices_l = np.random.permutation(x_tests_l.shape[0])

    x_tests_l = np.take(x_tests_l, indices_l, axis=0)
    y_tests_l = np.take(y_tests_l, indices_l, axis=0)

    rootLogger.info("Done.")

    return x_tests_l, y_tests_l

def gen_train_valid(EEG_dict_p, sex_dict_p, train_names_p, valid_names_p):

    x_trains_l = None
    y_trains_l = None

    x_valids_l = None
    y_valids_l = None


    ## Generate the train data
    rootLogger.info("Generatte the train data.")
    for name in train_names_p:

        eeg_l = EEG_dict_p[name]
        sex_l = sex_dict_p[name]        

        n_batch_l = np.shape(eeg_l)[1]
        sex_l = [sex_l] * n_batch_l

        if x_trains_l is None:
            x_trains_l = eeg_l
        else:        
            x_trains_l = np.concatenate((x_trains_l, eeg_l), axis=1)

        if y_trains_l is None:
            y_trains_l = sex_l
        else:
            y_trains_l = y_trains_l + sex_l

    x_trains_l = np.swapaxes(x_trains_l, 0, 1) # Make [batch, sample points, electrodes]

    indices_l = np.random.permutation(x_trains_l.shape[0])

    x_trains_l = np.take(x_trains_l, indices_l, axis=0)
    y_trains_l = np.take(y_trains_l, indices_l, axis=0)

    if valid_names_p is not None:
        ## Generate the validation data    
        rootLogger.info("Generatte the validation data.")
        for name in valid_names_p:

            eeg_l = EEG_dict_p[name]
            sex_l = sex_dict_p[name]        

            n_batch_l = np.shape(eeg_l)[1]
            sex_l = [sex_l] * n_batch_l

            if x_valids_l is None:
                x_valids_l = eeg_l
            else:        
                x_valids_l = np.concatenate((x_valids_l, eeg_l), axis=1)

            if y_valids_l is None:
                y_valids_l = sex_l
            else:
                y_valids_l = y_valids_l + sex_l

        x_valids_l = np.swapaxes(x_valids_l, 0, 1) # Make [batch, sample points, electrodes]            

        indices_l = np.random.permutation(x_valids_l.shape[0])

        x_valids_l = np.take(x_valids_l, indices_l, axis=0)
        y_valids_l = np.take(y_valids_l, indices_l, axis=0)    

    rootLogger.info("Done.")

    return x_trains_l, y_trains_l, x_valids_l, y_valids_l

class MLFlowCallback(keras.callbacks.Callback):

    def __init__(self, step_p=0):
        self.step = step_p

    def on_epoch_end(self, epoch, logs=None):
        try:
            mlflow.log_metric("loss", logs["loss"], step=self.step)
            mlflow.log_metric("accuracy", logs["accuracy"], step=self.step)
            mlflow.log_metric("val_loss", logs["val_loss"], step=self.step)
            mlflow.log_metric("val_accuracy", logs["val_accuracy"], step=self.step)
        except:
            pass

        self.step += 1

def test_run_n_epochs(exp_name_l, epochs_l, batch_size_l, nb_classes_l, train_valid_names_l, n_names_in_each_fold_l, EEG_dict_l, sex_dict_l, lr_l, valid_percentage_l, test_percentage_l, k_fold_cross_valid_l):

    i_loop = 0 # This option is to determine the number of epochs.

    rootLogger.info(f"============================== Loop {i_loop} ==============================")

    valid_names_l = [train_valid_names_l[i] for i in (np.arange(n_names_in_each_fold_l) + i_loop*n_names_in_each_fold_l)]
    train_names_l = [subj for subj in train_valid_names_l if subj not in valid_names_l]

    x_trains_l, y_trains_l, x_valids_l, y_valids_l = gen_train_valid(EEG_dict_l, sex_dict_l, train_names_l, valid_names_l)

    Y_train_l   = to_categorical(y_trains_l, nb_classes_l)               
    Y_val_l     = to_categorical(y_valids_l, nb_classes_l)               

    model_params_p = {}
    model_params_p["dropout_rate_l"] = 0.7139827978018466
    model_params_p["dropout_rate_FC_l"] = 0.4268363995888791
    model_params_p["reg_l2_l"] = 0.2707144605580773
    model_params_p["reg_l2_FC_l"] = 0.0531786212622587

    model_l, _, _ = loadModel(model_params_p)

    av_ml.AV_Model_compile(model_p=model_l, optimizer_p=keras.optimizers.Adam(lr_l), loss_p="categorical_crossentropy", metrics_p=["accuracy"])
    
    # early_stopping_l = EarlyStopping(patience=0, verbose=1)

    run_name_l = "test_run_n_epochs"

    del_run_id_by_name(exp_name_l, run_name_l)

    with mlflow.start_run(run_name=run_name_l) as new_run:

        history_l = av_ml.AV_Model_train_allbatches(    model_p=model_l,
                                                        x_p=x_trains_l,                 # Input data. It could be:
                                                                                        # A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
                                                                                        # A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
                                                                                        # A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample weights).
                                                                                        # None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
                                                        y_p=Y_train_l,                  # Target data. Like the input data x, it could be either Numpy array(s), framework-native tensor(s), list of Numpy arrays (if the model has multiple outputs) or None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays. If x is a generator, or keras.utils.Sequence instance, y should not be specified (since targets will be obtained from x).
                                                        batch_size_p=batch_size_l,      # Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of symbolic tensors, generators, or Sequence instances (since they generate batches).
                                                        epochs_p=epochs_l,              # Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
                                                        verbose_p=1,                    # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 
                                                        callbacks_p=[MLFlowCallback()],               # List of keras.callbacks.Callback instances. List of callbacks to apply during training and validation (if ). See callbacks.
                                                        validation_split_p=0.0,         # Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling. This argument is not supported when x is a generator or Sequence instance.
                                                        validation_data_p=(x_valids_l, Y_val_l),         # Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. validation_data will override validation_split. validation_data could be: - tuple (x_val, y_val) of Numpy arrays or tensors - tuple (x_val, y_val, val_sample_weights) of Numpy arrays - dataset or a dataset iterator
                                                                                        # For the first two cases, batch_size must be provided. For the last case, validation_steps must be provided.
                                                        shuffle_p=True,                 # Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
                                                        class_weight_p=None,            # Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
                                                        sample_weight_p=None,           # Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile(). This argument is not supported when x generator, or Sequence instance, instead provide the sample_weights as the third element of x.
                                                        initial_epoch_p=0,              # Integer. Epoch at which to start training (useful for resuming a previous training run).
                                                        steps_per_epoch_p=None,         # Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
                                                        validation_steps_p=None,        # Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.
                                                        validation_freq_p=1,            # Only relevant if validation data is provided. Integer or list/tuple/set. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a list, tuple, or set, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.
                                                        max_queue_size_p=10,            # Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
                                                        workers_p=1,                    # Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
                                                        use_multiprocessing_p=False     # Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.
                                                        )                

        mlflow.set_tag("mlflow.note.content", "This run is for estimating the good number of epochs.")
        mlflow.set_tag("Data", "Eye-Closed Resting EEG in the depression and mind-wandering projects.")

        mlflow.log_param("learning_rate", lr_l)
        mlflow.log_param("batch_size", batch_size_l)
        mlflow.log_param("num_epochs", epochs_l)        
        mlflow.log_param("Subjects for validation in percentage", valid_percentage_l)
        mlflow.log_param("Subjects for test in percentage", test_percentage_l)
        mlflow.log_param("The number of fold in k-Fold CV", k_fold_cross_valid_l)



def train():

    ## Set MLFlow
    _, _, exp_name_l = loadModel()
    set_mlflow_exp(exp_name_l)

    ## Train
    sex_dict_l = loadSex()
    EEG_dict_l = loadEEGs()

    sorted_names_l = sorted(EEG_dict_l.keys())

    nb_classes_l = 2 # The number of classs to be predicted.
    batch_size_l = 32
    epochs_l = 100
    lr_l = 0.001

    # k_fold_cross_valid_l = 3 # The number of folds in k-fold cross validation.
    k_fold_cross_valid_l = 10 # The number of folds in k-fold cross validation (>=2).
    valid_percentage_l = 20 # The percentage of subjects to be used as a validation set.
    test_percentage_l = 20 # The percentage of subjects to be used as a test set.

    n_test_l = int(np.floor(len(sorted_names_l)*test_percentage_l / 100))

    np.random.seed(0)
    random.seed(0)
    test_names_l = random.sample(sorted_names_l, n_test_l)
    train_valid_names_l = sorted([subj for subj in sorted_names_l if subj not in test_names_l])

    n_names_in_each_fold_l = int(np.floor(len(train_valid_names_l) / k_fold_cross_valid_l))
    n_names_k_fold_l = n_names_in_each_fold_l * k_fold_cross_valid_l
    train_valid_names_l = random.sample(train_valid_names_l, n_names_k_fold_l)
    
    rootLogger.info(f"kFold: {k_fold_cross_valid_l}, #Fold: {n_names_in_each_fold_l}, n train test: {len(train_valid_names_l)}, #Test subjects: {n_test_l}")

    x_test_l, y_test_l = gen_test(EEG_dict_l, sex_dict_l, test_names_l)
    Y_test_l = to_categorical(y_test_l, nb_classes_l)   

    if is_test_run_n_epochs:

        ## This function is to log the evolution of train in MLFlow.
        test_run_n_epochs(exp_name_l, epochs_l, batch_size_l, nb_classes_l, train_valid_names_l, n_names_in_each_fold_l, EEG_dict_l, sex_dict_l, lr_l, valid_percentage_l, test_percentage_l, k_fold_cross_valid_l)

    else:

        ## When we see potential of a model, we can explore other hyperparameters.
        kCV_valid_loss_l = []
        kCV_valid_acc_l = []

        kCV_test_loss_l = []
        kCV_test_acc_l = []

        step_l = 0;

        run_name_l = "run_hyperparameters"

        del_run_id_by_name(exp_name_l, run_name_l)

        with mlflow.start_run(run_name=run_name_l) as new_run:    

            for i_loop in range(k_fold_cross_valid_l):

                rootLogger.info(f"============================== Loop {i_loop} ==============================")

                valid_names_l = [train_valid_names_l[i] for i in (np.arange(n_names_in_each_fold_l) + i_loop*n_names_in_each_fold_l)]
                train_names_l = [subj for subj in train_valid_names_l if subj not in valid_names_l]

                x_trains_l, y_trains_l, x_valids_l, y_valids_l = gen_train_valid(EEG_dict_l, sex_dict_l, train_names_l, valid_names_l)

                Y_train_l   = to_categorical(y_trains_l, nb_classes_l)               
                Y_val_l     = to_categorical(y_valids_l, nb_classes_l)               
                
                model_l, _, _ = loadModel()

                av_ml.AV_Model_compile(model_p=model_l, optimizer_p=keras.optimizers.Adam(lr_l), loss_p="categorical_crossentropy", metrics_p=["accuracy"])
                
                # early_stopping_l = EarlyStopping(patience=0, verbose=1)

                history_l = av_ml.AV_Model_train_allbatches(    model_p=model_l,
                                                                x_p=x_trains_l,                 # Input data. It could be:
                                                                                                # A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
                                                                                                # A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
                                                                                                # A generator or keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample weights).
                                                                                                # None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
                                                                y_p=Y_train_l,                  # Target data. Like the input data x, it could be either Numpy array(s), framework-native tensor(s), list of Numpy arrays (if the model has multiple outputs) or None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays. If x is a generator, or keras.utils.Sequence instance, y should not be specified (since targets will be obtained from x).
                                                                batch_size_p=batch_size_l,      # Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32. Do not specify the batch_size if your data is in the form of symbolic tensors, generators, or Sequence instances (since they generate batches).
                                                                epochs_p=epochs_l,              # Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
                                                                verbose_p=1,                    # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 
                                                                # callbacks_p=[MLFlowCallback()],               # List of keras.callbacks.Callback instances. List of callbacks to apply during training and validation (if ). See callbacks.
                                                                validation_split_p=0.0,         # Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling. This argument is not supported when x is a generator or Sequence instance.
                                                                validation_data_p=(x_valids_l, Y_val_l),         # Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. validation_data will override validation_split. validation_data could be: - tuple (x_val, y_val) of Numpy arrays or tensors - tuple (x_val, y_val, val_sample_weights) of Numpy arrays - dataset or a dataset iterator
                                                                                                # For the first two cases, batch_size must be provided. For the last case, validation_steps must be provided.
                                                                shuffle_p=True,                 # Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
                                                                class_weight_p=None,            # Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
                                                                sample_weight_p=None,           # Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile(). This argument is not supported when x generator, or Sequence instance, instead provide the sample_weights as the third element of x.
                                                                initial_epoch_p=0,              # Integer. Epoch at which to start training (useful for resuming a previous training run).
                                                                steps_per_epoch_p=None,         # Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
                                                                validation_steps_p=None,        # Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.
                                                                validation_freq_p=1,            # Only relevant if validation data is provided. Integer or list/tuple/set. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a list, tuple, or set, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.
                                                                max_queue_size_p=10,            # Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
                                                                workers_p=1,                    # Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
                                                                use_multiprocessing_p=False     # Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.
                                                                )                


                score_valid_l = model_l.evaluate(x_valids_l, Y_val_l, verbose=0)
                print('Validate loss:', score_valid_l[0])
                print('Validate accuracy:', score_valid_l[1]) 

                kCV_valid_loss_l.append(score_valid_l[0])
                kCV_valid_acc_l.append(score_valid_l[1])


                score_test_l = model_l.evaluate(x_test_l, Y_test_l, verbose=0)
                print('Test loss:', score_test_l[0])
                print('Test accuracy:', score_test_l[1]) 

                kCV_test_loss_l.append(score_test_l[0])
                kCV_test_acc_l.append(score_test_l[1])


                mlflow.log_metric("Val_loss", score_valid_l[0], step=step_l)
                mlflow.log_metric("Val_accuracy", score_valid_l[1], step=step_l)
                mlflow.log_metric("Test_loss", score_test_l[0], step=step_l)
                mlflow.log_metric("Test_accuracy", score_test_l[1], step=step_l)

                step_l += 1        

            description_l = f"Val accuracy: mean {np.mean(kCV_valid_acc_l)}, SD {np.std(kCV_valid_acc_l, ddof=1)};\nVal loss: mean {np.mean(kCV_valid_loss_l)}, SD {np.std(kCV_valid_loss_l, ddof=1)}"
            mlflow.set_tag("mlflow.note.content", description_l)
            mlflow.set_tag("Data", "Eye-Closed Resting EEG in the depression and mind-wandering projects.")

            mlflow.log_param("learning_rate", lr_l)
            mlflow.log_param("batch_size", batch_size_l)
            mlflow.log_param("num_epochs", epochs_l)        
            mlflow.log_param("Subjects for validation in percentage", valid_percentage_l)
            mlflow.log_param("Subjects for test in percentage", test_percentage_l)
            mlflow.log_param("The number of fold in k-Fold CV", k_fold_cross_valid_l)


def prep_train_valid(EEG_dict_p, valid_name_p):

    valid_eeg_l = EEG_dict_p[valid_name_p]

    sorted_names_l = sorted(EEG_dict_p.keys())
    # print(sorted_names_l)

    train_eegs_l = None

    for train_name in sorted_names_l:
        if (train_name != valid_name_p):
            # print("Add: ", train_name)

            tmp_eeg_l = EEG_dict_p[train_name]
            if train_eegs_l is None:
                train_eegs_l = tmp_eeg_l
            else:                
                train_eegs_l = np.concatenate((train_eegs_l, EEG_dict_p[train_name]), axis=1)        

        else:
            print("Valid: ", valid_name_p)

    return train_eegs_l, valid_eeg_l

def loadSex():

    def _sex_code(txt):
        if txt == "หญิง":
            return female_code_g
        elif txt == "ชาย":
            return male_code_g
        else:
            raise Exception("Unmatched sex")

    ## Load data from MW project.
    dat_fn_l = 

    df_l = pd.read_excel(dat_fn_l, usecols="F,BC")
    df_l = df_l[df_l["Directory name"].notna()]

    # print(df)

    tmp_dict_sex_l = df_l.set_index("Directory name").to_dict(orient="index")

    dict_sex_l = {}
    for key, value in tmp_dict_sex_l.items():

        dict_sex_l[key] = _sex_code(value["ขอมลทวไป_เพศโดยกำเนด"])
    
    ## Load data from depression project.
    dat_fn_l = 
    
    doc_update_l = Document(dat_fn_l)
    sheet = doc_update_l.sheets[0]
    table = sheet.tables[0]
    rows_update_l = table.rows()    

    line_count = 0
    for row in rows_update_l:

        if line_count > 0: # Skip the table header        

            id_l = (table.cell(line_count, 1).value).strip()
            sex_l = (table.cell(line_count, 5).value).strip()

            dict_sex_l[id_l] = _sex_code(sex_l)

        line_count += 1

    return dict_sex_l

def loadEEGs():

    ## Load data from MW project.
    MW_exc_list_l = 

    dat_dir_l = 

    dir_list_l = os.listdir(dat_dir_l)

    EEG_dict_l = {}  

    for dir in dir_list_l:

        if dir not in MW_exc_list_l:

            eeg_path_l = os.path.join(dat_dir_l, dir, "SexPred", "EEG_1sec_epochs")

            if os.path.isfile(eeg_path_l):

                # rootLogger.info("Loading..." + eeg_path_l)

                with open(eeg_path_l, "rb") as fp:
                    input_l = pickle.load(fp)
                    fp.close()

                # print(np.shape(input_l))

                EEG_dict_l[dir] = input_l

    ## Load data from depression project.
    dat_dir_l = 

    dir_list_l = os.listdir(dat_dir_l)

    for dir in dir_list_l:
        
        eeg_path_l = os.path.join(dat_dir_l, dir, "SexPred", "EEG_1sec_epochs") 

        if os.path.isfile(eeg_path_l):

            # rootLogger.info("Loading..." + eeg_path_l)

            with open(eeg_path_l, "rb") as fp:
                input_l = pickle.load(fp)
                fp.close()

            EEG_dict_l[dir] = input_l


    return EEG_dict_l

def loadModel(model_params_p=None):

    # model_l, desc_l, exp_name_l = av_ml.AV_Machine_Conv1D_LeNetLikeV1(TS_LENGTH_p=256, TS_RGB_p=4, nb_classes_p=2, model_params_p=model_params_p)
    # model_l, desc_l, exp_name_l = av_ml.AV_Machine_Conv1D_Wang2016_ResNet(TS_LENGTH_p=256, TS_RGB_p=4, nb_classes_p=2, model_params_p=model_params_p)
    # model_l, desc_l, exp_name_l = av_ml.AV_Machine_Conv1D_vanPutten2018(TS_LENGTH_p=256, TS_RGB_p=4, nb_classes_p=2, model_params_p=model_params_p)
    # model_l, desc_l, exp_name_l = av_ml.AV_Machine_Conv1D_Zagoruyko2016_WideResNet(TS_LENGTH_p=256, TS_RGB_p=4, nb_classes_p=2, model_params_p=model_params_p)
    # model_l, desc_l, exp_name_l = av_ml.AV_Machine_Conv1D_Schirrmeister2017_shallowcovnet(TS_LENGTH_p=256, TS_RGB_p=4, nb_classes_p=2, model_params_p=model_params_p)
    model_l, desc_l, exp_name_l = av_ml.AV_Machine_Conv1D_Schirrmeister2017_deepcovnet(TS_LENGTH_p=256, TS_RGB_p=4, nb_classes_p=2, model_params_p=model_params_p)
    # model_l, desc_l, exp_name_l = av_ml.AV_Machine_Conv1D_Xception(TS_LENGTH_p=256, TS_RGB_p=4, nb_classes_p=2, model_params_p=model_params_p)

    return model_l, desc_l, exp_name_l

########## Test ##########
def test_link_mlflow_databricks():

    rootLogger.info("Logging Databricks Community Edition...")
    mlflow.login()

    rootLogger.info("Set tracking url...")
    mlflow.set_tracking_uri("databricks")

    mlflow.set_experiment("/check-databricks-connection")

    with mlflow.start_run():
        mlflow.log_metric("foo", 1)
        mlflow.log_metric("bar", 2)    

def test_mlflow_tf():
    # https://mlflow.org/docs/latest/deep-learning/tensorflow/quickstart/quickstart_tensorflow.html

    # Load the mnist dataset.
    train_ds, test_ds = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
    )

    # Let’s preprocess our data with the following steps: - Scale each pixel’s value to [0, 1). - Batch the dataset. - Use prefetch to speed up the training.   
    def preprocess_fn(data):
        image = tf.cast(data["image"], tf.float32) / 255
        label = data["label"]
        return (image, label)

    train_ds = train_ds.map(preprocess_fn).batch(128).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_fn).batch(128).prefetch(tf.data.AUTOTUNE)

    # Let’s define a convolutional neural network as our classifier. We can use keras.Sequential to stack up the layers.
    input_shape = (28, 28, 1)
    num_classes = 10

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )    

    # Set training-related configs, optimizers, loss function, metrics.
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(0.001),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )    

    exp_name = "mlflow-tf-keras-mnist"
    run_name = "test"

    if is_replace_run_g:
        del_run_id_by_name(exp_name, run_name)    

    set_mlflow_exp(exp_name)
    set_mlflow_run(run_name)

    mlflow.tensorflow.autolog()    



    # if run_id:
    #     rootLogger.info(f"Run ID for run name '{run_name}' is: {run_id}")
    #     # mlflow.delete_run(run_id)
    #     rootLogger.info(f"Run with ID {run_id} has been deleted.")
    # else:
    #     rootLogger.info(f"No run found with the name '{run_name}'.")

    # set_mlflow_exp("mlflow-tf-keras-mnist")
    # set_mlflow_run("test")

    

    # original_run_id = 'f3f716235c3348829ca79e04d78f82d9'
    # original_run = mlflow.get_run(original_run_id)
    # original_run_name = original_run.data.tags.get('mlflow.runName')

    # # with mlflow.start_run(run_name=original_run_name) as new_run:
    # with mlflow.start_run(run_id=original_run_id) as new_run:
        
    #     # for key, value in original_run.data.params.items():
    #     #     if key == 'REPLACE_WITH_YOUR_KEY':
    #     #         new_value = REPLACE_WITH_YOUR_VALUE
    #     #         mlflow.log_param(key, new_value)
    #     #     else:
    #     #         mlflow.log_param(key, value)

    #     # for key, value in original_run.data.metrics.items():
    #     #     mlflow.log_metric(key, value)    

    model.fit(x=train_ds, epochs=3)

def test_muse_models():

    # model_l, _, _ = av_ml.AV_Machine_Conv1D_LeNetLikeV1(TS_LENGTH_p=256, TS_RGB_p=4, nb_classes_p=2, model_params_p="")
    # model_l, _, _ = av_ml.AV_Machine_Conv1D_Wang2016_ResNet(TS_LENGTH_p=256, TS_RGB_p=4, nb_classes_p=2, model_params_p="")
    model_l, _, _ = av_ml.AV_Machine_Conv1D_vanPutten2018(TS_LENGTH_p=256, TS_RGB_p=4, nb_classes_p=2, model_params_p="")    
    # model_l, _, _ = av_ml.AV_Machine_Conv1D_Zagoruyko2016_WideResNet(TS_LENGTH_p=256, TS_RGB_p=4, nb_classes_p=2, model_params_p="")
    # model_l, _, _ = av_ml.AV_Machine_Conv1D_Schirrmeister2017_shallowcovnet(TS_LENGTH_p=256, TS_RGB_p=4, nb_classes_p=2, model_params_p="")
    # model_l, _, _ = av_ml.AV_Machine_Conv1D_Schirrmeister2017_deepcovnet(TS_LENGTH_p=256, TS_RGB_p=4, nb_classes_p=2, model_params_p="")
    # model_l, _, _ = av_ml.AV_Machine_Conv1D_Xception(TS_LENGTH_p=256, TS_RGB_p=4, nb_classes_p=2, model_params_p="")

    model_l.summary()


if __name__ == "__main__":
    # This comment is for testing dagshub.
    start_time_l = time.ctime()

    rootLogger.info("Start")

    # test_link_mlflow_databricks()
    # link_mlflow_databricks()
    # link_mlflow_local()

    # test_mlflow_tf()

    test_muse_models()

    # train()

    # hype_opt()    
    # hype_opt_vis()
    # hype_opt_best_network()

    
    rootLogger.info("Stop")

    plt.show()
