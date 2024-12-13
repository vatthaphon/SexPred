import json
import logging
import mlflow
import numpy as np
import os
import pandas as pd
import pickle
import random
import shutil
import sys
import time

from fastapi import FastAPI, Request
from mlflow.tracking import MlflowClient
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from numbers_parser import Document
from pydantic import BaseModel
from tensorflow.keras.utils import to_categorical

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

########## Tweaked parameters ##########
is_replace_run_g = True

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

def get_model_path(run_id_p, artifact_path_p="model"):

    ## Initialize the MLflow client
    client_l = MlflowClient()

    ## Construct the full model path
    model_path_l = client_l.download_artifacts(run_id_p, artifact_path_p) # This method downloads the model into the Temp directory.

    rootLogger.info(f"Model Path: {model_path_l}")

    return model_path_l

def get_model(run_id_p):

    ## Download the model from MLflow server to temp directory.
    model_path_l = get_model_path(run_id_p)

    rootLogger.info(f"Model was downloaded to {model_path_l}")

    ## Copy the model to the "model" of the current working directory.
    save_model_dir_l = "model"

    if os.path.exists("./" + save_model_dir_l):        
        shutil.rmtree("./" + save_model_dir_l)
        rootLogger.info("Delete the old model.")
    else:
        rootLogger.info(f"Couldn't find ./{save_model_dir_l}.")

    shutil.copytree(model_path_l, "./" + save_model_dir_l)

    ## Restore the model from the path.
    model_l = mlflow.tensorflow.load_model("./" + save_model_dir_l)

    ## Get test accuracy of the model.
    sex_dict_l = loadSex()
    EEG_dict_l = loadEEGs()

    sorted_names_l = sorted(EEG_dict_l.keys())

    nb_classes_l = 2 # The number of classs to be predicted.

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

    x_test_l, y_test_l = gen_test(EEG_dict_l, sex_dict_l, test_names_l)
    Y_test_l = to_categorical(y_test_l, nb_classes_l)   

    score_valid_l = model_l.evaluate(x_test_l, Y_test_l, verbose=0)
    valid_acc = score_valid_l[1]
    rootLogger.info(f"Validate accuracy: {valid_acc}") 

    return model_l

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

########## Test ##########

if __name__ == "__main__":
    ## This commemt is added for testing dagshub.
    start_time_l = time.ctime()

    rootLogger.info("Start")

    # link_mlflow_databricks()
    link_mlflow_local()

    run_id_l = "30787f6373ae4b59917bf0c3388db868"
    get_model(run_id_l)
    
    rootLogger.info("Stop")
