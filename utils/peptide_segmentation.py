import logging
import h5py
import os
import numpy as np
import pandas as pd
import raw_signal_utils
import dask.bag as db
from dask.diagnostics import ProgressBar

ProgressBar().register()


def compute_fractional_blockage(scaled_raw, open_pore):
    scaled_raw = np.array(scaled_raw, dtype=float)
    scaled_raw /= open_pore
    scaled_raw = np.clip(scaled_raw, a_max=1., a_min=0.)
    return scaled_raw


def find_peptides(signal, voltage, signal_threshold=0.7,
                  voltage_threshold=-180):
    diff_points = np.where(np.abs(np.diff(np.where(
        np.logical_and(voltage <= voltage_threshold,
                       signal <= signal_threshold), 1, 0))) == 1)[0]
    if voltage[0] <= voltage_threshold and signal[0] <= signal_threshold:
        diff_points = np.hstack([[0], diff_points])
    if voltage[-1] <= voltage_threshold and signal[-1] <= signal_threshold:
        diff_points = np.hstack([diff_points, [len(voltage)]])

    return zip(diff_points[::2], diff_points[1::2])


def _find_peptides_helper(raw_signal_meta,
                          voltage=None, open_pore_prior=220,
                          open_pore_prior_stdv=35, signal_threshold=0.7,
                          voltage_threshold=-180, min_duration_obs=0,
                          voltage_change_delay=3):
    run, channel, raw_signal = raw_signal_meta
    peptide_metadata = []
    open_pore = raw_signal_utils.find_open_pore_current(
        raw_signal, open_pore_guess=open_pore_prior,
        bound=open_pore_prior_stdv)
    if open_pore is None:
        open_pore = open_pore_prior
    frac_signal = compute_fractional_blockage(raw_signal, open_pore)
    peptide_segments = find_peptides(frac_signal, voltage,
                                     signal_threshold=signal_threshold,
                                     voltage_threshold=voltage_threshold)
    for peptide_segment in peptide_segments:
        if peptide_segment[1] - peptide_segment[0] - voltage_change_delay \
           < min_duration_obs:
            continue
        peptide_start = peptide_segment[0] + voltage_change_delay
        peptide_end = peptide_segment[1]
        peptide_signal = frac_signal[peptide_start:peptide_end]
        peptide_metadata.append((run, channel, peptide_start, peptide_end,
                                 peptide_end - peptide_start,
                                 np.mean(peptide_signal),
                                 np.std(peptide_signal),
                                 np.median(peptide_signal),
                                 np.min(peptide_signal),
                                 np.max(peptide_signal),
                                 open_pore))
    return peptide_metadata


def parallel_find_peptides(f5_fnames, good_channel_dict, open_pore_prior,
                           open_pore_prior_stdv, signal_threshold,
                           voltage_threshold, min_duration_obs,
                           save_location=".",
                           save_prefix="segmented_peptides",
                           voltage_change_delay=3,
                           n_workers=1):
    logger = logging.getLogger("parallel_find_peptides")
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    for run, f5_fname in f5_fnames.iteritems():
        logger.info("Reading in signals for run: %s" % run)
        f5 = h5py.File(f5_fname)
        voltage = f5.get("/Device/MetaData").value["bias_voltage"] * 5.
        good_channels = good_channel_dict.get(run)
        raw_signals = []

        for channel_no in good_channels:
            channel = "Channel_%s" % str(channel_no)
            logger.debug(channel)
            raw_signal = raw_signal_utils.get_scaled_raw_for_channel(
                f5, channel=channel)
            raw_signals.append((run, channel, raw_signal))

        logger.debug("Loading up the bag with signals.")
        bag = db.from_sequence(raw_signals, npartitions=128)
        peptide_map = bag.map(
            _find_peptides_helper, voltage=voltage,
            open_pore_prior=open_pore_prior,
            open_pore_prior_stdv=open_pore_prior_stdv,
            signal_threshold=signal_threshold,
            voltage_threshold=voltage_threshold,
            min_duration_obs=min_duration_obs,
            voltage_change_delay=voltage_change_delay)
        logger.debug("Running peptide segmenter.")
        peptide_metadata_by_channel = \
            peptide_map.compute(num_workers=n_workers)
        logger.debug("Converting list of peptides to a dataframe.")
        # peptide_metadata_by_channel = list(peptide_metadata_by_channel)
        peptide_metadata = []
        while len(peptide_metadata_by_channel) > 0:
            peptide_metadata.extend(peptide_metadata_by_channel.pop())
        peptide_metadata_df = pd.DataFrame.from_records(
            peptide_metadata,
            columns=["run", "channel", "start_obs", "end_obs", "duration_obs",
                     "mean", "stdv", "median", "min", "max", "open_channel"])
        save_name = save_prefix + "_%s.pkl" % (run)
        try:
            os.makedirs(save_location)
        except OSError:
            pass
        logger.debug("Saving dataframe to pickle.")
        peptide_metadata_df.to_pickle(os.path.join(save_location, save_name))


def extract_raw_data(f5_fnames, df_location=".",
                     df_prefix="segmented_peptides",
                     save_location=".",
                     save_prefix="segmented_peptides_raw_data",
                     open_pore_prior=220.,
                     open_pore_prior_stdv=35.):
    logger = logging.getLogger("extract_raw_data")
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    for run, f5_fname in f5_fnames.iteritems():
        logger.info("Saving data from %s" % run)
        df = np.load(os.path.join(df_location, df_prefix + "_%s.pkl" % run),
                     allow_pickle=True)
        f5 = h5py.File(f5_fname, "r")
        peptides = []
        last_channel = None
        for i, row in df.iterrows():
            if row.channel != last_channel:
                last_channel = row.channel
                raw_signal = raw_signal_utils.get_scaled_raw_for_channel(
                    f5, channel=row.channel)
                if "open_channel" in row.index:
                    logger.debug("Attempting to get open channel from peptide "
                                 "df.")
                    open_pore = row.open_channel
                else:
                    logger.debug("Attempting to find open channel current.")
                    open_pore = raw_signal_utils.find_open_pore_current(
                        raw_signal, open_pore_guess=open_pore_prior,
                        bound=open_pore_prior_stdv)
                    if open_pore is None:
                        logger.debug("Open channel couldn't be found, using "
                                     "the given prior.")
                        open_pore = open_pore_prior
                open_pore = np.floor(open_pore)

                logger.debug("Computing fractional current.")
                frac_signal = compute_fractional_blockage(raw_signal,
                                                          open_pore)

            peptide_signal = frac_signal[row["start_obs"]:row["end_obs"]]
            logger.debug("Mean in df: %0.4f, \tMean in extracted: %0.4f" %
                         (row["mean"], np.mean(peptide_signal)))
            logger.debug("Len in df: %d, \tLen in extracted: %d" %
                         (row["duration_obs"], len(peptide_signal)))
            peptides.append(peptide_signal)
        logger.debug("Saving to file.")
        assert len(df) == len(peptides)
        np.save(os.path.join(save_location, save_prefix + "_%s.npy" % run),
                peptides)
