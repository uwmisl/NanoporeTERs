import sys
sys.path.append('./pore_utils')
import numpy as np
import pandas as pd
import h5py
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from yaml_assistant import YAMLAssistant


# # Finding Regions with Normal Voltage
# Returns tuple list of open voltage regions (between flips)

def find_peptide_voltage_changes(voltage, voltage_threshold=-180):
    diff_points = np.where(np.abs(np.diff(np.where(
        voltage <= voltage_threshold, 1, 0))) == 1)[0]
    if voltage[0] <= voltage_threshold:
        diff_points = np.hstack([[0], diff_points])
    if voltage[-1] <= voltage_threshold:
        diff_points = np.hstack([diff_points, [len(voltage)]])

    return zip(diff_points[::2], diff_points[1::2])


# # Retrieving Related Data Files from Filtered File or Capture File
# Returns list containing path names of `[raw_file, capture_file, config_file]` corresponding to the passed in file name. Passed in file can be either filter file or capture file (total captures).

def get_related_files (file_name):
    date = os.path.basename(file_name)[0:8] # e.g. "20190603"
    run_seg = file_name[-11:-4] # e.g. "run01_a"
    
    raw_file_dir = "/disk1/pore_data/MinION_raw_data_%s" % date
    raw_file = [x for x in os.listdir(raw_file_dir) if run_seg in x][0]
    
    capture_file_dir = "/disk1/pore_data/segmented/peptides/%s" % date
    if file_name.endswith(".csv"):
        # Given file is the filtered file and we're looking for the capture file
        filtered_file = file_name
        capture_file = [x for x in os.listdir(capture_file_dir) if x.endswith(run_seg + ".pkl")][0]
    elif file_name.endswith(".pkl"):
        # Given file is the capture file and filtered file is unspecified
        capture_file = file_name
        filtered_file = "Unspecified"
    else:
        print("Invalid file name")
        return
    
    config_file_dir = "/ssd1/home/kz9/peptide-signatures/configs"
    possible_configs = [x for x in os.listdir(config_file_dir) if date in x and not "backup" in x]
    config_file = None
    for config in possible_configs:
        run_segs_in_config = YAMLAssistant(os.path.join(config_file_dir, config)).get_variable("fast5:names").keys()
        if run_seg in run_segs_in_config:
            config_file = config
    
    print ("Filter File: " + filtered_file)
    raw_file = os.path.join(raw_file_dir, raw_file)
    print ("Raw File: " + raw_file)
    capture_file = os.path.join(capture_file_dir, capture_file)
    print ("Capture File: " + capture_file)
    config_file = os.path.join(config_file_dir, config_file)
    print ("Config File: " + config_file)
    print
    
    return [raw_file, capture_file, config_file]


# # Finding Capture Times for One Channel
# Returns list of all capture times from a single channel. Called by `get_capture_time` and `get_capture_time_tseg`

def calc_time_until_capture (voltage_changes, segment_starts, segment_ends, blockage_starts, blockage_ends):
    all_capture_times = []
    
    # If there are no captures for the entire channel
    if not segment_starts:
        return None
    
    capture_time = 0
    voltage_ix = 0
    segment_ix = 0
    blockage_ix = 0
    while voltage_ix < len(voltage_changes) and segment_ix < len(segment_starts):
        voltage_start, voltage_end = voltage_changes[voltage_ix]
        
        # Check if voltage region contains captures
        if segment_starts[segment_ix] < voltage_end:
            
            # Point of reference for calculating capture time. For first capture in voltage region, this will be the
            # start time of the voltage region. For subsequent captures it will be the end time of the previous capture.
            start_time = voltage_start 
            
            # Loop through captures within the current voltage region
            while segment_ix < len(segment_starts) and segment_starts[segment_ix] < voltage_end:
                segment_start = segment_starts[segment_ix]
                segment_end = segment_ends[segment_ix]

                if segment_start >= voltage_start:  
                    
                    # Don't count times when channel is blocked from other junk
                    # Subtracts all blockages that have occurred since the last capture
                    while blockage_starts[blockage_ix] < segment_start:
                        capture_time -= blockage_ends[blockage_ix] - blockage_starts[blockage_ix]
                        blockage_ix += 1 

                    # Add capture time to list & reset stuff
                    all_capture_times.append(segment_start - start_time + capture_time)
                    start_time = segment_end
                    segment_ix += 1
                    capture_time = 0
                    blockage_ix += 1 # Skip blockage caused by current segment

                # Something weird is happening... report it and move on
                else:
                    print (voltage_start, voltage_end)
                    print (segment_start)
                    segment_ix += 1
            
            # Last capture in current voltage region sets up capture time for first capture in next voltage region
            capture_time = voltage_end - segment_end  

        # If no captures in current voltage region, extend capture time for duration of region
        else:
            capture_time += voltage_end - voltage_start
            
        voltage_ix += 1

    return all_capture_times


# Find time between all blockages:

def calc_time_until_capture_blockages (voltage_changes, blockage_starts, blockage_ends):
    time_until_capture = []
    
    # If there are no captures for the entire channel
    if not blockage_starts:
        return None
    
    blockage_extend = 0
    voltage_ix = 0
    blockage_ix = 0
    first = True
    while voltage_ix < len(voltage_changes) and blockage_ix < len(blockage_starts):
        voltage_start, voltage_end = voltage_changes[voltage_ix]
        blockage_start = blockage_starts[blockage_ix]
        blockage_end = blockage_ends[blockage_ix]

        # Check for a blockage in the region
        if blockage_start < voltage_end: 
            # If there's a blockage start between the voltage start & end...
            if blockage_start >= voltage_start:
                # If this is the first blockage in this voltage region
                if first:
                    time_until_capture.append(blockage_start - voltage_start + blockage_extend)
                    blockage_extend = 0
                    blockage_ix += 1
                    first = False
                else:  
                    time_until_capture.append(blockage_start - blockage_ends[blockage_ix - 1])
                    blockage_ix += 1
            # Blockage was captured before this voltage region, so move on to next blockage
            else:
                # print voltage_start, voltage_end
                # print blockage_start
                blockage_ix += 1
        # If blockage is no longer in the region
        else:
            # If there actually aren't any blockages in this region
            if first:
                blockage_extend += voltage_end - voltage_start
                voltage_ix += 1
            # If this is due to the fact that there were multiple blockages in the previous region
            else: 
                blockage_extend += voltage_end - blockage_ends[blockage_ix - 1]
                voltage_ix += 1
                first = True

    return time_until_capture


# # Getting Time Between Captures
# Returns list of average times until capture for given time intervals of a single run

def get_time_between_captures (filtered_file, time_interval=None):

    # Retrieve raw file, unfiltered capture file, and config file names
    files = get_related_files(filtered_file)
    raw_file = files[0]
    capture_file = files[1]
    config_file = files[2]
    
    # Process raw file
    f5 = h5py.File(raw_file)
    # Find regions where voltage is normal
    voltage = f5.get("/Device/MetaData").value["bias_voltage"] * 5.
    voltage_changes = find_peptide_voltage_changes(voltage)
    f5.close()
    
    # Process unfiltered captures file
    if capture_file.endswith(".pkl"):
        blockages = pd.read_pickle(capture_file)
    else:
        blockages = pd.read_csv(capture_file, index_col=0, header=0, sep="\t")
        
    # Set of strings (i.e. "Channel_511") of all good channels in the run (no repeats) 
    # good_channels = set(blockages.channel) 
    
    # Process config file
    y = YAMLAssistant(config_file)
    good_channels = y.get_variable("fast5:good_channels:" + filtered_file[-11:-4]) # [-11:-4] gives run_seg (i.e. "run01_a")
    for i in range(0, len(good_channels)):
        good_channels[i] = "Channel_" + str(good_channels[i])
    print ("Number of Channels: " + str(len(good_channels)))
    print
    
    # Process filtered captures file
    captures = pd.read_csv(filtered_file, index_col=0, header=0, sep="\t")
    
    # Break run into time segments based on time interval (given in minutes).
    # If no time interval given then just take average time between captures of entire run
    if time_interval:
        time_segments = range(time_interval * 60 * 10000, len(voltage) + 1, time_interval * 60 * 10000)
    else:
        time_segments = [len(voltage)]
        
        
    ## Calculate Average Time Between Captures for Each Time Segment ##
    
    time_elapsed = [0 for x in range(0, len(good_channels))] # Tracks time elapsed (no capture) for each channel across time segments
    timepoint_captures = [] # List of mean capture times across all channels for each timepoint
    captures_count = [] # Number of captures across all channels for each timepoint
    checkpoint = 0
    for timepoint in time_segments:
        voltage_changes_segment = []
        # Find open voltage regions that start within this time segment
        for voltage_region in voltage_changes:
            if voltage_region[0] < timepoint and voltage_region[0] >= checkpoint:
                voltage_changes_segment.append(voltage_region)
        # If this time segment contains open voltage regions...
        if voltage_changes_segment:
            end_voltage_seg = voltage_changes_segment[len(voltage_changes_segment) - 1][1] # End of last voltage region in tseg
            capture_times = [] # Master list of all capture times from this tseg
            # Loop through all good channels and get captures times from each
            for i, channel in enumerate(good_channels): 
                channel_blockages = blockages[blockages.channel == channel]
                blockage_exists = False
                # If there are any blockages in this tseg (includes both non-captures and captures)
                if not channel_blockages.empty and not channel_blockages[np.logical_and(channel_blockages.start_obs <= end_voltage_seg, channel_blockages.start_obs > checkpoint)].empty:
                    blockage_exists = True
                    blockage_segment = channel_blockages[np.logical_and(channel_blockages.start_obs <= end_voltage_seg, channel_blockages.start_obs > checkpoint)]
                    blockage_starts = list(blockage_segment.start_obs)
                    blockage_ends = list(blockage_segment.end_obs)
                    
                channel_captures = captures[captures.channel == channel]
                # Check that channel actually has captures in this tseg
                if not channel_captures.empty and not channel_captures[np.logical_and(channel_captures.start_obs <= end_voltage_seg, channel_captures.start_obs > checkpoint)].empty:
                    captures_segment = channel_captures[np.logical_and(channel_captures.start_obs <= end_voltage_seg, channel_captures.start_obs > checkpoint)]
                    segment_starts = list(captures_segment.start_obs)
                    segment_ends = list(captures_segment.end_obs)
                    
                    time_until_capture = calc_time_until_capture(voltage_changes_segment, segment_starts, segment_ends, blockage_starts, blockage_ends)
                    # Add time since channel's last capture from previous tsegs to time until first capture in current tseg
                    time_until_capture[0] += time_elapsed[i]
                     
                    # Update to new time elapsed (time from end of last capture in this tseg to end of tseg minus blockages)
                    time_elapsed[i] = 0
                    voltage_ix = 0
                    while voltage_ix < len(voltage_changes_segment):
                        if voltage_changes_segment[voltage_ix][0] > segment_ends[-1]:
                            time_elapsed[i] += np.sum(calc_time_until_capture_blockages(voltage_changes_segment[voltage_ix:], blockage_starts, blockage_ends))
                            break
                        voltage_ix += 1
                    time_elapsed[i] += end_voltage_seg - blockage_ends[-1]
                    
                    capture_times.extend(time_until_capture)
                else:
                    # No captures but still blockages, so add duration of open voltage regions minus blockages to time elapsed
                    if blockage_exists:
                        time_elapsed[i] += np.sum(calc_time_until_capture_blockages(voltage_changes_segment, blockage_starts, blockage_ends))
                        time_elapsed[i] += end_voltage_seg - blockage_ends[-1]
                    # No captures or blockages for channel in this tseg, so add total duration of open voltage regions to time elapsed
                    else:
                        time_elapsed[i] += np.sum([voltage_region[1] - voltage_region[0] for voltage_region in voltage_changes_segment])
            if capture_times:
                timepoint_captures.append(np.mean(capture_times))
            else:
                timepoint_captures.append(-1)
            
            # Plot histogram of capture times
#             sns.distplot(capture_times)
#             plt.xlabel("Time Between Captures (obs)")
#             plt.ylabel("Density")
#             plt.show()
            
            captures_count.append(len(capture_times))
            checkpoint = end_voltage_seg
        else:
            print ("No open voltage region in time segment [" + str(checkpoint) + ", " + str(timepoint) + "]")
            print
            timepoint_captures.append(-1)
            checkpoint = timepoint
    
    print ("Number of Captures: " + str(captures_count))
    print
    return timepoint_captures


# # Getting Capture Frequency
# Returns list of capture frequencies (captures/channel/min) for each time interval.
# Time intervals must be equal duration and start from zero!

def get_capture_freq (filtered_file, time_interval=None):

    # Retrieve raw file, unfiltered capture file, and config file names
    files = get_related_files(filtered_file)
    raw_file = files[0]
    capture_file = files[1]
    config_file = files[2]
    
    # Process raw file
    f5 = h5py.File(raw_file)
    # Find regions where voltage is normal
    voltage = f5.get("/Device/MetaData").value["bias_voltage"] * 5.
    voltage_changes = find_peptide_voltage_changes(voltage)
    f5.close()
    
    # Process unfiltered captures file
    if capture_file.endswith(".pkl"):
        blockages = pd.read_pickle(capture_file)
    else:
        blockages = pd.read_csv(capture_file, index_col=0, header=0, sep="\t")
    
    # Process config file
    y = YAMLAssistant(config_file)
    good_channels = y.get_variable("fast5:good_channels:" + filtered_file[-11:-4]) # [-11:-4] gives run_seg (i.e. "run01_a")
    for i in range(0, len(good_channels)):
        good_channels[i] = "Channel_" + str(good_channels[i])
    print ("Number of Channels: " + str(len(good_channels)))
    print
    
    # Process filtered captures file
    captures = pd.read_csv(filtered_file, index_col=0, header=0, sep="\t")
    
    # Break run into time segments based on time interval (given in minutes).
    # If no time interval given then just take average time between captures of entire run
    if time_interval:
        time_segments = range(time_interval * 60 * 10000, len(voltage) + 1, time_interval * 60 * 10000)
    else:
        time_segments = [len(voltage)]
        
        
    ## Calculate Capture Frequency for Each Time Segment ##
    
    all_capture_freq = [] # List of capture frequencies for each timepoint
    checkpoint = 0
    for timepoint in time_segments:
        voltage_changes_segment = []
        # Find open voltage regions that start within this time segment
        for voltage_region in voltage_changes:
            if voltage_region[0] < timepoint and voltage_region[0] >= checkpoint:
                voltage_changes_segment.append(voltage_region)
                
        # If this time segment contains open voltage regions...
        if voltage_changes_segment:
            end_voltage_seg = voltage_changes_segment[len(voltage_changes_segment) - 1][1] # End of last voltage region in tseg
            capture_counts = [] # List of capture counts for each channel from this tseg (length of list = # of channels)
            # Loop through all good channels and get captures times from each
            for i, channel in enumerate(good_channels): 
                channel_captures = captures[captures.channel == channel]
                # Check that channel actually has captures and add the # of captures in this tseg to capture_counts
                if not channel_captures.empty:
                    capture_counts.append(len(channel_captures[np.logical_and(channel_captures.start_obs <= end_voltage_seg, channel_captures.start_obs > checkpoint)]))
                else:
                    capture_counts.append(0)
            all_capture_freq.append(np.mean(capture_counts) / (time_segments[0] / 600000.))
            checkpoint = end_voltage_seg
        else:
            print ("No open voltage region in time segment [" + str(checkpoint) + ", " + str(timepoint) + "]")
            print
            all_capture_freq.append(0)
            checkpoint = timepoint
    
    return all_capture_freq


# # Calibration Curves

def NTER_time_fit(time):
    if time == -1:
        return 0
    conc = np.power(time / 20384, 1 / -0.96)
    if conc < 0:
        return 0
    return conc

def NTER_freq_fit(freq):
    if freq == -1:
        return 0
    conc = np.power(freq / 1.0263, 1 / 0.5239)
    if conc < 0:
        return 0
    return conc

