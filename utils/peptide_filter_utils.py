import numpy as np
import h5py
import peptide_quantifier_utils as pepquant
import joblib
import os
from NTERs_trained_cnn_05152019 import *
import pandas as pd
import logging
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
use_cuda = True


def check_capture_rejection(end_capture, voltage_ends, tol_obs=20):
    for voltage_end in voltage_ends:
        if np.abs(end_capture - voltage_end) < tol_obs:
            return True
    return False


def get_num_classes(classifier, classifier_name):
    if classifier_name == "NTER_cnn":
        return classifier.fc2.out_features
    elif classifier_name == "NTER_rf":
        return len(classifier.classes_)
    else:
        return


# Possible classifier names: NTER_cnn, NTER_rf
# Prediction classes are 1-9:
# 0:Y00, 1:Y01, 2:Y02, 3:Y03, 4:Y04, 5:Y05, 6:Y06, 7:Y07, 8:Y08, 9:noise,
# -1:below conf_thesh
def init_classifier(classifier_name, classifier_path):
    if classifier_name is "NTER_cnn":
        # CNN classifier
        nanoporeTER_cnn = load_cnn(classifier_path)
        nanoporeTER_cnn.eval()
        return nanoporeTER_cnn
    elif classifier_name is "NTER_rf":
        # Random Forest classifier
        return joblib.load(open(classifier_path, 'rb'))
    else:
        raise Exception("Invalid classifier name")


# Possible filter names: "NTER_general"
def get_filter_param(filter_name):
    # What filter param each value in the output array represents:
    # [mean_low, mean_high, stdv_high, med_low, med_high, min_low, min_high,
    # max_low, max_high, length, fname_ext]
    if filter_name is "NTER_general":
        return [0, 0.45, 1, 0.15, 1, 0.005, 1, 0, 0.65, 20100, ""]
    else:
        raise Exception("Invalid filter name")


def print_param(filter_param):
    s = ""
    s += "Mean: " + str((filter_param[0], filter_param[1])) + "\n"
    s += "Stdv: " + str((0, filter_param[2])) + "\n"
    s += "Median: " + str((filter_param[3], filter_param[4])) + "\n"
    s += "Min: " + str((filter_param[5], filter_param[6])) + "\n"
    s += "Max: " + str((filter_param[7], filter_param[8])) + "\n"
    s += "Length: " + str(filter_param[9]) + "\n"
    return s


# Returns -1 if classification probability is below confidence threshold
def classifier_predict(classifier, raw, conf_thresh, classifier_name):
    if classifier_name == "NTER_cnn":
        X_test = np.array([raw])
        # go from 2D to 3D array (each obs in a capture becomes its own array)
        X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
        X_test = X_test[:, :19881]  # take only first 19881 obs of each capture
        # break all obs in a captures into 141 groups of 141 (19881 total); each
        # capture becomes its own array
        X_test = X_test.reshape(len(X_test), 1, 141, 141)
        X_test = torch.from_numpy(X_test)
        X_test = X_test.cuda()
        outputs = classifier(X_test)
        out = nn.functional.softmax(outputs)
        prob, lab = torch.topk(out, 1)
        if prob < conf_thresh:
            return -1
        lab = lab.cpu().numpy()
        return lab[0][0]
    else:
        class_proba = classifier.predict_proba(
            [[np.mean(raw), np.std(raw), np.min(raw), np.max(raw),
              np.median(raw)]])[0]
        max_proba = np.amax(class_proba)
        if max_proba >= conf_thresh:
            return np.where(class_proba == max_proba)[0][0]
        return -1


# date is a string
# runs is a list of strings, i.e. ["run01_a", "run01_b"]
# filter_name can only be "NTER_general" until more types of filters as added
# classifier_name can be "NTER_cnn" or "NTER_rf" for CNN and Random Forest
#   classifiers respectively
# conf_thresh is confidence threshold for classifiers; only classifications >=
#   conf_thresh will be written to file
# custom_fname is a custom string to be added to file name
# rej_check ensures that captures which are ejected prematurely are not counted
def filter_and_classify_peptides(runs, date, filter_name, classifier_name="",
                                 conf_thresh=0.95,
                                 custom_fname="", rej_check=True,
                                 f5_dir="", classifier_path="",
                                 capture_fname="", raw_fname="", save_dir="."):

    logger = logging.getLogger("filter_and_classify_peptides")
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    filter_param = get_filter_param(filter_name)
    if custom_fname:
        filter_param[10] = filter_param[10] + custom_fname + "_"

    logger.info("Params for " + filter_name + " Filter:")
    logger.info(print_param(filter_param))

    if classifier_name:
        classifier = init_classifier(classifier_name, classifier_path)
        logger.info("Confidence Threshold: " + str(conf_thresh))

    all_filtered_files = []

    for run in runs:
        logger.info("Starting run chunk " + run)

        # Prep filenames
        capture_file = capture_fname % run
        raw_file = raw_fname % run

        # TODO parameterize
        f5_file = os.path.join(f5_dir,
                               [x for x in os.listdir(f5_dir) if
                                run in x][0])
        logger.debug("f5_file:" + f5_file)
        logger.debug("raw_file:" + raw_file)
        logger.debug("capture_file:" + capture_file)

        # Read data into variables
        capture_meta_df = pd.read_pickle(capture_file)
        raw_captures = np.load(raw_file, allow_pickle=True)
        f5 = h5py.File(f5_file, "r")

        # Get the voltage & where it switches
        voltage = f5.get("/Device/MetaData").value["bias_voltage"] * 5.
        voltage_changes = pepquant.find_peptide_voltage_changes(voltage)
        voltage_ends = [x[1] for x in voltage_changes]

        # Apply length filter
        capture_meta_df = capture_meta_df[capture_meta_df.duration_obs >
                                          filter_param[9]]

        # Apply 5 feature filters and classify
        if classifier_name:
            captures = [[] for x in range(0, get_num_classes(classifier,
                                                             classifier_name))]
        else:
            captures = [[]]
        non_filtered = 0
        non_classified = 0
        for i in capture_meta_df.index:
            # To keep track of filter progress
            if i % 100 == 0 and i != 0:
                logger.debug(str(i))

            meta_i = capture_meta_df.loc[i, :]

            # If capture is ejected early, don't count it
            if rej_check:
                capture_rejected = check_capture_rejection(
                    meta_i.end_obs, voltage_ends)
                if not capture_rejected:
                    continue

            raw_minus_10 = raw_captures[i][10:]  # skip first 10 obs of capture

            new_mean = np.mean(raw_minus_10)
            new_med = np.median(raw_minus_10)
            new_min = np.min(raw_minus_10)
            new_max = np.max(raw_minus_10)
            new_stdv = np.std(raw_minus_10)

            capture = [i, meta_i["run"], meta_i["channel"], meta_i["start_obs"],
                       meta_i["end_obs"], meta_i["duration_obs"]]

            if (new_mean > filter_param[0] and new_mean < filter_param[1] and
                new_stdv < filter_param[2] and new_med > filter_param[3] and
                new_med < filter_param[4] and new_min > filter_param[5] and
                new_min < filter_param[6] and new_max > filter_param[7] and
                    new_max < filter_param[8]):
                meta_i["mean"] = new_mean
                meta_i["median"] = new_med
                meta_i["min"] = new_min
                meta_i["max"] = new_max
                capture.extend([new_mean, new_stdv, new_med,
                                new_min, new_max, meta_i["open_channel"]])

                if classifier_name:
                    # classifier uses obs 100-20100 of capture
                    raw_100_to_20100 = raw_captures[i][100:20100]
                    class_predict = classifier_predict(classifier,
                                                       raw_100_to_20100,
                                                       conf_thresh,
                                                       classifier_name)
                    if class_predict == -1:
                        non_classified += 1
                    else:
                        captures[class_predict].append(capture)
                else:
                    captures[0].append(capture)
            else:
                non_filtered += 1

        no_pass = float(non_filtered) / len(capture_meta_df.index) * 100
        logger.info("Summary:")
        logger.info("Did not pass filter: %0.3f %%" % no_pass)
        if classifier_name:
            semi_pass = float(non_classified) / len(capture_meta_df.index) * 100
            logger.info("Passed filter but not classifier: %0.3f %%" %
                        semi_pass)

        # Save filtered captures. If classifier was enabled, each class is a
        # different file.
        for i, class_captures in enumerate(captures):
            if class_captures:
                filtered_captures = pd.DataFrame(class_captures)
                filtered_captures.index = filtered_captures[0]
                del filtered_captures[0]
                filtered_captures.columns = capture_meta_df.columns

                if "cnn" in classifier_name:
                    filtered_fname = "%s_segmented_peptides_filtered%s_cnn_class%02d_%s.csv" % (date, filter_param[10], i, run)
                    filtered_fname = os.path.join(save_dir, filtered_fname)

                elif "rf" in classifier_name:
                    filtered_fname = "%s_segmented_peptides_filtered%s_rf_class%02d_%s.csv" % (date, filter_param[10], i, run)
                    filtered_fname = os.path.join(save_dir, filtered_fname)
                else:
                    filtered_fname = "%s_segmented_peptides_filtered%s_%s" % \
                                     (date, filter_param[10], run)
                    filtered_fname = os.path.join(save_dir, filtered_fname)
                logger.info("Saving to " + filtered_fname)
                filtered_captures.to_csv(filtered_fname, sep="\t", index=True)
                all_filtered_files.append(filtered_fname)

        del captures
        f5.close()
        torch.cuda.empty_cache()

    return all_filtered_files
