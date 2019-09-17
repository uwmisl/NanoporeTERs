import numpy as np
import h5py
import peptide_quantifier_utils as pepquant
from sklearn.externals import joblib
import os
from NTER_CNN import *
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
use_cuda = True


def check_capture_rejection(end_capture, voltage_ends, tol_obs=20):
    for voltage_end in voltage_ends:
        if np.abs(end_capture - voltage_end) < tol_obs:
            return True
    return False


def get_num_classes(classifier):
    if isinstance(classifier, CNN):
        return 10 # NTER_cnn classifier has 10 classes
    else:
        return len(classifier.classes_)


# Possible classifier names: NTER_cnn, NTER_rf
# Prediction classes are 1-9:
# 0:Y00, 1:Y01, 2:Y02, 3:Y03, 4:Y04, 5:Y05, 6:Y06, 7:Y07, 8:Y08, 9:noise, -1:below conf_thesh
def init_classifier(classifier_name):
    if classifier_name is "NTER_cnn":
        # CNN classifier trained on Y00-Y08_plusnoise
        classifier_path = "/disk1/pore_data/jeff_saved/NTERs_trained_cnn_05152019.pt"
        print "Classifier: " + classifier_path
        nanoporeTER_cnn = load_CNN(classifier_path)
        nanoporeTER_cnn.eval()
        return nanoporeTER_cnn
    elif classifier_name is "NTER_rf":
        # Random Forest classifier
        classifier_path = "/disk1/pore_data/NanoporeTERs/Y00_Y08_Noise_03082019_RandomForest_model.sav"
        print "Classifier: " + classifier_path
        return joblib.load(open(classifier_path, 'rb'))
    else:
        raise Exception("Invalid classifier name")


# Possible filter names: "NTER_general"
def get_filter_param(filter_name):
    # What filter param each value in the output array represents:
    # [mean_low, mean_high, stdv_high, med_low, med_high, min_low, min_high, max_low, max_high, length, fname_ext]
    if filter_name is "NTER_general":
        return [0, 0.45, 1, 0.15, 1, 0.005, 1, 0, 0.65, 20100, ""]
    else:
        raise Exception("Invalid filter name")
        
        
def print_param(filter_param):
    print "Mean: " + str((filter_param[0], filter_param[1]))
    print "Stdv: " + str((0, filter_param[2]))
    print "Median: " + str((filter_param[3], filter_param[4]))
    print "Min: " + str((filter_param[5], filter_param[6]))
    print "Max: " + str((filter_param[7], filter_param[8]))
    print "Length: " + str(filter_param[9])


# Returns -1 if classification probability is below confidence threshold
def classifier_predict(classifier, raw, conf_thresh):
    if isinstance(classifier, CNN):
        X_test = np.array([raw])
        X_test = X_test.reshape(len(X_test), X_test.shape[1], 1) # go from 2D to 3D array (each obs in a capture becomes its own array)
        X_test = X_test[:,:19881] # take only first 19881 obs of each capture
        X_test = X_test.reshape(len(X_test),1,141,141) # break all obs in a captures into 141 groups of 141 (19881 total); each capture becomes its own array
        X_test = torch.from_numpy(X_test)
        X_test = X_test.cuda()
        outputs = classifier(X_test)
        out = nn.functional.softmax(outputs)
        prob, lab = torch.topk(out,1)
        if prob < conf_thresh:
            return -1
        lab = lab.cpu().numpy()
        return lab[0][0]
    else:
        class_proba = classifier.predict_proba([[np.mean(raw), np.std(raw), np.min(raw), np.max(raw), np.median(raw)]])[0]
        max_proba = np.amax(class_proba)
        if max_proba >= conf_thresh:
            return np.where(class_proba == max_proba)[0][0]
        return -1


# date is a string
# runs is a list of strings, i.e. ["run01_a", "run01_b"]
# filter_name can only be "NTER_general" until more types of filters as added
# classifier_name can be "NTER_cnn" or "NTER_rf" for CNN and Random Forest classifiers respectively
# conf_thresh is confidence threshold for classifiers; only classifications >= conf_thresh will be written to file
# custom_fname is a custom string to be added to file name
# rej_check ensures that captures which are ejected prematurely are not counted
def filter_and_classify_peptides(date, runs, filter_name, classifier_name="", conf_thresh=0.95, 
                                 custom_fname="", rej_check=True):
    
    filter_param = get_filter_param(filter_name)
    print "Params for " + filter_name + " Filter:"
    print_param(filter_param) 
    print
    
    if classifier_name: 
        classifier = init_classifier(classifier_name)
        print "Confidence Threshold: " + str(conf_thresh)
        print
    
    segmented_base_fname = "/disk1/pore_data/segmented/peptides/%s/%s_segmented_peptides_%s%s%s.%s" 
    
    all_filtered_files = []
    
    for run in runs:
        print "Starting run chunk", run

        # Prep filenames
        capture_file = segmented_base_fname % (date, date, "", "", run, "pkl") 
        raw_file = segmented_base_fname % (date, date, "raw_data_", "", run, "npy") 
        f5_file_dir = "/disk1/pore_data/MinION_raw_data_%s" % date
        f5_file = os.path.join(f5_file_dir, [x for x in os.listdir(f5_file_dir) if run in x][0])
        print f5_file
        print raw_file
        print capture_file

        # Read data into variables
        capture_meta_df = pd.read_pickle(capture_file) 
        raw_captures = np.load(raw_file)
        f5 = h5py.File(f5_file, "r")

        # Get the voltage & where it switches
        voltage = f5.get("/Device/MetaData").value["bias_voltage"] * 5.
        voltage_changes = pepquant.find_peptide_voltage_changes(voltage)
        voltage_ends = [x[1] for x in voltage_changes]

        # Apply length filter
        capture_meta_df = capture_meta_df[capture_meta_df.duration_obs > filter_param[9]]

        # Apply 5 feature filters and classify
        if classifier_name:
            captures = [[] for x in range(0, get_num_classes(classifier))]
        else:
            captures = [[]]
        non_filtered = 0
        non_classified = 0
        for i in capture_meta_df.index:
            # To keep track of filter progress
            if i % 100 == 0 and i != 0:
                print i

            meta_i = capture_meta_df.loc[i, :]

            # If capture is ejected early, don't count it 
            if rej_check:
                capture_rejected = check_capture_rejection(meta_i.end_obs, voltage_ends)
                if not capture_rejected:
                    continue

            raw_minus_10 = raw_captures[i][10:] # skip first 10 obs of capture

            new_mean = np.mean(raw_minus_10)
            new_med = np.median(raw_minus_10)
            new_min = np.min(raw_minus_10)
            new_max = np.max(raw_minus_10)
            new_stdv = np.std(raw_minus_10)

            capture = [i, meta_i["run"], meta_i["channel"], meta_i["start_obs"], meta_i["end_obs"],
                       meta_i["duration_obs"]]

            if (new_mean > filter_param[0] and new_mean < filter_param[1] and new_stdv < filter_param[2] and new_med > filter_param[3] 
                and new_med < filter_param[4] and new_min > filter_param[5] and new_min < filter_param[6] and new_max > filter_param[7] 
                and new_max < filter_param[8]):
                meta_i["mean"] = new_mean
                meta_i["median"] = new_med
                meta_i["min"] = new_min
                meta_i["max"] = new_max
                capture.extend([new_mean, new_stdv, new_med, new_min, new_max, meta_i["open_channel"]])

                if classifier_name:
                    raw_100_to_20100 = raw_captures[i][100:20100] # classifier uses obs 100-20100 of capture
                    class_predict = classifier_predict(classifier, raw_100_to_20100, conf_thresh)
                    if class_predict == -1:
                        non_classified += 1
                    else:
                        captures[class_predict].append(capture)
                else:
                    captures[0].append(capture)
            else:
                non_filtered += 1

        print "% did not pass filter (out of total captures): " + str(float(non_filtered) / len(capture_meta_df.index) * 100)
        if classifier_name:
            print "% passed filter but not classifier (out of total captures): " + str(float(non_classified) / len(capture_meta_df.index) * 100)

        # Save filtered captures. If classifier was enabled, each class is a different file.
        if custom_fname:
            filter_param[10] = filter_param[10] + custom_fname + "_"
        for i, class_captures in enumerate(captures):
            if class_captures:
                filtered_captures = pd.DataFrame(class_captures)
                filtered_captures.index = filtered_captures[0]
                del filtered_captures[0]
                filtered_captures.columns = columns=capture_meta_df.columns

                if "cnn" in classifier_name:
                    filtered_fname = segmented_base_fname % (date, date, "filtered_", filter_param[10] + "cnn_class%02d_" % i, run, "csv")
                elif "rf" in classifier_name:
                    filtered_fname = segmented_base_fname % (date, date, "filtered_", filter_param[10] + "rf_class%02d_" % i, run, "csv")
                else: 
                    filtered_fname = segmented_base_fname % (date, date, "filtered_", filter_param[10], run, "csv")                    
                print "Saving to", filtered_fname
                filtered_captures.to_csv(filtered_fname, sep="\t", index=True)
                all_filtered_files.append(filtered_fname)

        del captures
        f5.close()
        torch.cuda.empty_cache()
        
    return all_filtered_files