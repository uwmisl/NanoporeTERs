import h5py
import numpy as np
from matplotlib import pyplot as plt
import re
import time
import logging
import subprocess
from shutil import copyfile
from yaml_assistant import *

_raw_data_paths = {
    "read-based": {
        "Signal": "/Raw/Reads/Read_%d/Signal",
        "UniqueGlobalKey": "/UniqueGlobalKey",
        "Meta": "/UniqueGlobalKey/channel_id",
        "multi": False
    },
    "channel-based": {
        "Signal": "/Raw/Channel_%d/Signal",
        "UniqueGlobalKey": "/UniqueGlobalKey",
        "Meta": "/Raw/Channel_%d/Meta",
        "multi": True
    }
}


def natkey(string_):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', string_)]


def compute_fractional_blockage(scaled_raw, open_pore):
    '''Compute fractional blockage from the scaled raw data. Result is in the range
    [0,1].

    Compute the fraction of the pore that is "blocked" by the captured
    sample, where 0 is a completely open pore and 1 is completely blocked (no
    current can pass through the captured sample.

    scale_raw_current or get_scaled_raw_for_channel must be called first.
    '''
    scaled_raw = np.array(scaled_raw)
    f = np.vectorize(_calc_frac, otypes=[np.float])
    frac = f(scaled_raw, open_pore)
    return frac


def _calc_frac(x, open_pore):
    '''Helper fn for vectorizing fractional blockage calculations.'''
    return min([1., max([0., 1. - x / open_pore])])


def get_fractional_blockage(f5, open_pore_guess=220, open_pore_bound=15,
                            channel=None, read=None):
    '''Retrieve the scaled raw signal for the channel, compute the open pore
    current, and return the fractional blockage for that channel.'''
    signal = get_scaled_raw_for_channel(f5, channel=channel, read=read)
    open_pore = find_open_pore_current(signal, open_pore_guess,
                                       bound=open_pore_bound)
    if open_pore is None:
        return None
    frac = compute_fractional_blockage(signal, open_pore)
    return frac


def get_local_fractional_blockage(f5, open_pore_guess=220, open_pore_bound=15,
                                  channel=None, read=None,
                                  local_window_sz=1000):
    '''Retrieve the scaled raw signal for the channel, compute the open pore
    current, and return the fractional blockage for that channel.'''
    signal = get_scaled_raw_for_channel(f5, channel=channel, read=read)
    open_pore = find_open_pore_current(signal, open_pore_guess,
                                       bound=open_pore_bound)
    if open_pore is None:
        print "open pore is None"

        return None

    frac = np.zeros(len(signal))
    for start in range(0, len(signal), local_window_sz):
        end = start + local_window_sz
        local_chunk = signal[start: end]
        local_open_channel = find_open_pore_current(local_chunk, open_pore,
                                                    bound=open_pore_bound)
        if local_open_channel is None:
            local_open_channel = open_pore
        frac[start:end] = compute_fractional_blockage(local_chunk,
                                                      local_open_channel)
    return frac


def get_sampling_rate(f5):
    try:
        sample_rate = f5.get("Meta").attrs['sample_rate']
        return sample_rate
    except AttributeError:
        pass
    try:
        file_data = f5.get("UniqueGlobalKey")
        sample_rate = int(file_data.get("context_tags")
                          .attrs["sample_frequency"])
        return sample_rate
    except AttributeError:
        pass
    raise ValueError("Cannot find sample rate.")


def get_raw_signal(f5, channel=None):
    if channel is not None:
        channel_no = int(re.findall(r"(\d+)", channel)[0])
        signal_path = _raw_data_paths.get("channel-based") \
                                     .get("Signal") % channel_no
        raw = f5.get(signal_path).value
        return raw
    else:
        raw = f5.get("/Raw/Reads/").values()[0].get("Signal").value
        return raw


def determine_f5_format(f5):
    raw_base_path = str(f5.get("/Raw").values()[0])
    if "Reads" in raw_base_path:
        path_type = "read-based"
    elif "Channel" in raw_base_path:
        path_type = "channel-based"
    else:
        path_type = None
    return path_type


def get_scale_metadata(f5, path_type=None, channel=None, read=None):
    if not path_type:
        path_type = determine_f5_format(f5)
    if read is not None and path_type == "read-based":
        meta_path = _raw_data_paths.get(path_type).get("Meta")
    elif channel is not None and path_type == "channel-based":
        channel_no = int(re.findall(r"(\d+)", channel)[0])
        meta_path = _raw_data_paths.get(path_type).get("Meta") % channel_no
    else:
        raise ValueError("Need to specify either read or channel, and it needs"
                         " to match the internal file structure.")
    attrs = f5.get(meta_path).attrs
    offset = attrs.get("offset")
    rng = attrs.get("range")
    digi = attrs.get("digitisation")
    return offset, rng, digi


def get_scaled_raw_for_channel(f5, channel=None, read=None):
    '''Note: using UK sp. of digitization for consistency w/ file format'''
    if channel is not None:
        path_type = "channel-based"
    else:
        path_type = "read-based"
    raw = get_raw_signal(f5, channel=channel)
    offset, rng, digi = get_scale_metadata(f5, path_type, channel=channel,
                                           read=read)
    return scale_raw_current(raw, offset, rng, digi)


def scale_raw_current(raw, offset, rng, digitisation):
    '''Note: using UK sp. of digitization for consistency w/ file format'''
    return (raw + offset) * (rng / digitisation)


def find_open_pore_current(raw, open_pore_guess, bound=None):
    if bound is None:
        bound = 0.1 * open_pore_guess
    upper_bound = open_pore_guess + bound
    lower_bound = open_pore_guess - bound
    ix_in_range = np.where(np.logical_and(raw <= upper_bound,
                                          raw > lower_bound))[0]
    if len(ix_in_range) == 0:
        open_pore = None
    else:
        open_pore = np.median(raw[ix_in_range])
    return open_pore


def find_signal_off_regions(raw, window_sz=200, slide=100, current_range=50):
    off = []
    for start in range(0, len(raw), slide):
        window_mean = np.mean(raw[start:start + window_sz])
        if window_mean < np.abs(current_range) \
           and window_mean > -np.abs(current_range):
            off.append(True)
        else:
            off.append(False)
    off_locs = np.multiply(np.where(off)[0], slide)
    loc = None
    if len(off_locs) > 0:
        last_loc = off_locs[0]
        start = last_loc
        regions = []
        for loc in off_locs[1:]:
            if loc - last_loc != slide:
                regions.append((start, last_loc))
                start = loc
            last_loc = loc
        if loc is not None:
            regions.append((start, loc))
        return regions
    else:
        return []


def find_high_regions(raw, window_sz=200, slide=100, open_pore=1400,
                      current_range=300):
    off = []
    for start in range(0, len(raw), slide):
        window_mean = np.mean(raw[start:start + window_sz])
        if window_mean > (open_pore + np.abs(current_range)):
            off.append(True)
        else:
            off.append(False)
    off_locs = np.multiply(np.where(off)[0], slide)
    regions = []

    if len(off_locs) > 0:
        loc = None
        last_loc = off_locs[0]
        start = last_loc

        for loc in off_locs[1:]:
            if loc - last_loc != slide:
                regions.append((start, last_loc + window_sz))
                start = loc
            last_loc = loc
        if loc is not None:
            regions.append((start, loc + window_sz))
    return regions


def sec_to_obs(time_in_sec, sample_rate_hz):
    return time_in_sec * sample_rate_hz


def split_multi_fast5(yml_file, temp_f5_fname=None):
    logger = logging.getLogger("split_multi_fast5")
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    y = YAMLAssistant(yml_file)
    f5_dir = y.get_variable("fast5:dir")
    prefix = y.get_variable("fast5:prefix")
    names = y.get_variable("fast5:names")
    run_splits = y.get_variable("fast5:run_splits")

    new_names = {}

    for i, (run, name) in enumerate(names.iteritems()):
        f5_fname = f5_dir + "/" + prefix + name
        f5 = h5py.File(f5_fname, "r")
        sample_freq = get_sampling_rate(f5)
        splits = run_splits.get(run)

        # Prep file to write to
        if temp_f5_fname is None:
            temp_f5_fname = "temp.fast5"

        # Copy the original fast5 file
        logger.info("Preparing a template fast5 file for %s splits." % run)
        copyfile(src=f5_fname, dst=temp_f5_fname + ".")
        temp_f5 = h5py.File(temp_f5_fname + ".")

        # Delete the contents
        logger.debug("Deleting its contents.")
        try:
            del temp_f5["/IntermediateData"]
        except KeyError:
            logger.debug("No /IntermediateData in %s" %
                         f5.filename)
            pass
        try:
            del temp_f5["/MultiplexData"]
        except KeyError:
            logger.debug("No /MultiplexData in %s" % f5.filename)
            pass
        try:
            del temp_f5["/Device/MetaData"]
        except KeyError:
            logger.error("No /Device/MetaData in %s"
                         % f5.filename)
            pass
        try:
            del temp_f5["/StateData"]
        except KeyError:
            logger.debug("No /StateData in %s" % f5.filename)
            pass
        for channel_no in range(1, 513):
            channel = "Channel_%d" % channel_no
            try:
                del temp_f5["/Raw/%s/Signal" % channel]
            except KeyError:
                logger.debug("No /Raw/%s/Signal in %s" %
                             (channel, f5.filename))
                pass
        temp_f5.flush()
        temp_f5.close()
        subprocess.call(["h5repack", "-f", "GZIP=1", temp_f5_fname + ".",
                         temp_f5_fname])
        os.remove(temp_f5_fname + ".")
        open_split_f5 = {}

        logger.info("Copying the template for each split and adding metadata.")
        for split in splits:
            run_split = run + "_" + split.get("name")
            split_f5_fname = f5_dir + "/" + prefix + run_split + ".temp.fast5"

            try:
                os.remove(split_f5_fname)
            except OSError:
                pass

            new_names[run_split] = run_split + ".fast5"
            logger.info("Split: %s" % run_split)
            logger.debug(split_f5_fname)
            copyfile(src=temp_f5_fname, dst=split_f5_fname)

            split_f5 = h5py.File(split_f5_fname)

            # Save the file handle to the dict
            open_split_f5[run_split] = split_f5

            # Write the metadata to the file
            split_start_sec = split.get("start")
            split_end_sec = split.get("end")
            split_start_obs = sec_to_obs(split_start_sec, sample_freq)
            split_end_obs = sec_to_obs(split_end_sec, sample_freq)

            metadata = f5.get("/Device/MetaData").value
            metadata_segment = metadata[split_start_obs:split_end_obs]
            split_f5.create_dataset("/Device/MetaData",
                                    metadata_segment.shape,
                                    dtype=metadata_segment.dtype)
            split_f5["/Device/MetaData"][()] = metadata_segment

        os.remove(temp_f5_fname)

        logger.info("Splitting fast5, processing one channel at a time.")
        for channel_no in range(1, 513):
            channel = "Channel_%d" % channel_no
            logger.info("    %s" % channel)
            raw = get_raw_signal(f5, channel=channel)
            
            for split in splits:
                run_split = run + "_" + split.get("name")
                logger.debug(run_split)
                # Get timing info
                split_start_sec = split.get("start")
                split_end_sec = split.get("end")
                split_start_obs = sec_to_obs(split_start_sec, sample_freq)
                split_end_obs = sec_to_obs(split_end_sec, sample_freq)

                # Extract the current segment
                segment = raw[split_start_obs:split_end_obs]

                # Save to the fast5 file
                split_f5 = open_split_f5[run_split]
                logger.debug(split_f5.filename)
                split_f5.create_dataset(
                    "/Raw/Channel_%d/Signal" % (channel_no), (len(segment),),
                    dtype='int16')
                split_f5["/Raw/Channel_%d/Signal" % (channel_no)][()] = segment
                split_f5.flush()

        logger.info("Closing and compressing files.")
        for run_split, split_f5 in open_split_f5.iteritems():
            logger.debug(run_split)
            split_f5_temp_name = split_f5.filename
            split_f5.close()
            split_f5_fname = f5_dir + "/" + prefix + run_split + ".fast5"
            logger.debug("Saving as %s" % split_f5_fname)
            subprocess.call(["h5repack", "-f", "GZIP=1",
                            split_f5_temp_name,
                            split_f5_fname])
            os.remove(split_f5_temp_name)

    archive_fname = yml_file.split(".")
    archive_fname.insert(-1, "backup.%s" % time.strftime("%Y%m%d-%H%M"))
    archive_fname = os.path.abspath(".".join(archive_fname))
    logger.info("Backing up the config file to %s" % archive_fname)
    copyfile(os.path.abspath(yml_file), archive_fname)

    logger.info("Saving new filenames to config file.")
    y.write_variable("fast5:names", new_names)
    y.write_variable("fast5:original_names", names)
    logger.info("Done")


def judge_channels(fast5_fname, plot_grid=False, cmap=None,
                   expected_open_pore=235):
    '''Judge channels based on quality of current. If the current is too
    low, the channel is probably off (bad), etc.'''
    if cmap is None:
        cmap = make_cmap([(0.02, 0.02, 0.02),
                          (0.7, 0.7, 0.7),
                          (0.98, 0.98, 1)])
    f5 = h5py.File(name=fast5_fname)
    channels = f5.get("Raw").keys()
    channels.sort(key=natkey)
    nrows, ncols = 16, 32  # = 512 channels
    channel_grid = np.zeros((nrows, ncols))
    if plot_grid:
        fig, ax = plt.subplots(figsize=(16, 32))
    for channel in channels:
        i = int(re.findall(r'Channel_(\d+)', channel)[0])
        row_i = (i - 1) / ncols
        col_j = (i - 1) % ncols

        raw = get_scaled_raw_for_channel(f5, channel)

        # Case 1: Channel might not be totally off, but has little variance
        if np.abs(np.mean(raw)) < 20 and np.std(raw) < 50:
            if plot_grid:
                ax.text(col_j, row_i, str(i), va='center', ha='center',
                        color='white')
            continue

        # Case 2: Neither the median or 75th percentile value contains the
        #         open pore current.
        if expected_open_pore is not None:
            sorted_raw = sorted(raw)
            len_raw = len(raw)
            q_50 = len_raw / 2
            q_75 = len_raw * 3 / 4
            median_outside_rng = np.abs(sorted_raw[q_50] - expected_open_pore) > 25
            upper_outside_rng = np.abs(sorted_raw[q_75] - expected_open_pore) > 25
            if median_outside_rng and upper_outside_rng:
                channel_grid[row_i, col_j] = 0.5
                if plot_grid:
                    ax.text(col_j, row_i, str(i), va='center', ha='center',
                            color='white')
                continue

        # Case 3: The channel is off
        off_regions = find_signal_off_regions(raw, current_range=100)
        off_points = []
        for start, end in off_regions:
            off_points.extend(range(start, end))
        if len(off_points) + 50000 > len(raw):
            if plot_grid:
                ax.text(col_j, row_i, str(i), va='center', ha='center',
                        color='white')
            continue
            
        # Case 4: The channel is assumed to be good
        channel_grid[row_i, col_j] = 1
        if plot_grid:
            ax.text(col_j, row_i, str(i), va='center', ha='center',
                    color='black')
    if plot_grid:
        ax.imshow(channel_grid, cmap=cmap)
        ax.set_xticks(np.arange(-.5, ncols, 1))
        ax.set_yticks(np.arange(-.5, nrows, 1))
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.grid(color='white', linestyle='-', linewidth=10)

    good_channels = np.add(np.where(channel_grid.flatten() == 1), 1).flatten()
    return channel_grid, good_channels


def plot_channel_grid(channel_grid, cmap,
                      title=None,
                      colorbar=False,
                      cbar_minmax=(0, None),
                      grid_kwargs={'color': 'white',
                                   'linestyle': '-',
                                   'linewidth': 10}):
    '''Simple version of judge_channels that makes a channel visualization
    plot given an arbitrary channel grid.'''
    fig, ax = plt.subplots(figsize=(16, 32))
    cbar_min, cbar_max = cbar_minmax
    if cbar_min is None:
        cbar_min = np.min(channel_grid)
    if cbar_max is None:
        cbar_max = np.max(channel_grid)
    im = ax.imshow(channel_grid, cmap=cmap,
                   vmin=cbar_min, vmax=cbar_max)
    channel_no = 1
    z = np.max(channel_grid)
    for i in range(channel_grid.shape[0]):
        for j in range(channel_grid.shape[1]):
            label_color = cmap(channel_grid[i, j] / z)
            if np.mean(label_color[:3]) < 0.5:
                text_color = 'white'
            else:
                text_color = 'black'
            ax.text(j, i, str(channel_no), va='center', ha='center',
                    color=text_color)
            channel_no += 1
    ax.set_xticks(np.arange(-.5, channel_grid.shape[1], 1))
    ax.set_yticks(np.arange(-.5, channel_grid.shape[0], 1))
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.grid(**grid_kwargs)
    if colorbar:
        pos = fig.add_axes([0.185, 0.39, 0.65, 0.01])
        plt.colorbar(im, orientation="horizontal", cax=pos)
    if title is not None:
        ax.set_title(title)
    return fig, ax

import sys


def make_cmap(colors, position=None, bit=False):
    '''
    Source: http://schubert.atmos.colostate.edu/~cslocum/custom_cmap.html
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0, 1, 256)
    if position is None:
        position = np.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return cmap
