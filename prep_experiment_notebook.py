#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
import string
import os
import re
alpha = list(string.ascii_lowercase)


# # Change the variables in this cell

# In[2]:


date = "20190504"
f5_base_dir = "/disk1/pore_data"
f5_dir = "MinION_raw_data_%s" % date


# In[3]:


min_duration_obs = 10
signal_threshold = 0.7
open_pore_mean = 220.
open_pore_stdv = 35.


# In[4]:


try:
    os.makedirs(os.path.join(f5_base_dir, f5_dir))
except OSError:
    pass


# In[5]:


for fname in os.listdir(f5_base_dir):
    if date in fname and fname.endswith(".fast5"):
        mv_cmd = "".join(["mv ", os.path.join(f5_base_dir, fname), " ", os.path.join(f5_base_dir, f5_dir)]) + "/"
        print mv_cmd
        get_ipython().system('{mv_cmd}')


# # Functions for reading Google Drive spreadsheet

# In[6]:


def import_gdrive_sheet(gdrive_key, sheet_id):
    run_spreadsheet = pd.read_csv("https://docs.google.com/spreadsheet/ccc?key=" +                                   peptide_gdrive_key + "&output=csv&gid=" + sheet_id)
    run_spreadsheet.Date = pd.to_datetime(run_spreadsheet.Date, format="%m_%d_%y")
    return run_spreadsheet

peptide_gdrive_key = "1CRnphJXZ4QZSg21-0SlcyUwO0H1nMGTfTjH_G7q9fJM"
sheet_id = "1709785742"

run_spreadsheet = import_gdrive_sheet(peptide_gdrive_key, sheet_id)


# In[7]:


def get_run_info(run_spreadsheet, date_yyyymmdd, runs=None):
    date = datetime.date(int(date_yyyymmdd[:4]), int(date_yyyymmdd[4:6]), int(date_yyyymmdd[6:]))
    all_runs = run_spreadsheet[["Date", "File name"]].drop_duplicates()
    runs_on_date_ix = []
    for i, run_date, run_fname in all_runs.itertuples():
        if not isinstance(run_fname, str):
            continue
        fname_search = re.findall(r"(\d+)_(\d+)_(\d+)_(run_\d+)", run_fname)
        if len(fname_search) == 0 or len(fname_search[0]) < 4:
            continue
        m, d, y, run = fname_search[0]
        if len(y) == 2:
            y = "20" + y
        fname_date = datetime.date(int(y), int(m), int(d))
        if fname_date == date:
            runs_on_date_ix.append(i)
    runs_on_date = all_runs.loc[runs_on_date_ix]
    runs_on_date["Date"] = date
    
    if runs is not None:
        runs_on_date = runs_on_date[runs_on_date["File name"].isin(runs)]
    
    runs_by_date = {}
    for i in runs_on_date.index:
        start_line = i
        next_ix = list(all_runs.index).index(start_line) + 1
        if next_ix >= len(all_runs.index):
            end_line = run_spreadsheet.index[-1]
        else:
            end_line = list(all_runs.index)[next_ix] - 1
        runs_by_date[runs_on_date.loc[i, "File name"]] = run_spreadsheet.loc[start_line:end_line, :]

    formatted_coords = {}
    for run, df in runs_by_date.iteritems():
        formatted_coords[run] = []
        # print df
        for i, coords in enumerate(df.loc[:, ["start (sec)", "end (sec)"]].iterrows()):
            letter = alpha[i]
            if np.isnan(coords[1][0]):
                continue
            start = int(coords[1][0])
            if np.isnan(coords[1][1]):
                end = start + 100
            else:
                end = int(coords[1][1])
            formatted_coords[run].append({"name": letter, "start": start, "end": end})
            
    return runs_by_date


# # Sort runs by flow cell

# In[8]:


runs_by_date = get_run_info(run_spreadsheet, date)


# In[9]:


runs_by_date_df = runs_by_date.values()
flowcells = []
for df in runs_by_date_df:
    if df.iloc[0]["Flow Cell"] not in flowcells:
         flowcells.append(df.iloc[0]["Flow Cell"])


# In[10]:


all_f5_files = [x for x in os.listdir(os.path.join(f5_base_dir, f5_dir)) if x.endswith(".fast5")]


# In[11]:


f5_files_by_flowcell = dict.fromkeys(flowcells)
for flowcell in flowcells:
    f5_files_by_flowcell[flowcell] = [f5 for f5 in all_f5_files if flowcell in f5]


# In[12]:


print f5_files_by_flowcell


# In[13]:


prefixes_by_flowcell = dict.fromkeys(flowcells)
for flowcell in flowcells:
    if f5_files_by_flowcell[flowcell]:
        prefixes_by_flowcell[flowcell] = re.findall(r"(.*_)run_\d+_\d+.fast5", f5_files_by_flowcell[flowcell][0])[0]


# In[14]:


print prefixes_by_flowcell


# # Generate config file(s)

# Separate config files are generated for runs with separate flow cells.

# In[15]:


config_files_by_flowcell = dict.fromkeys(flowcells)
for flowcell in flowcells:
    if f5_files_by_flowcell[flowcell]:
        config_files_by_flowcell[flowcell] = "configs/segment_%s_%s.yml" % (date, flowcell)


# ## Print example config file(s)

# In[16]:


for flowcell in flowcells:
    if f5_files_by_flowcell[flowcell]:
        print "fast5:"
        print "  dir: %s/" % os.path.join(f5_base_dir, f5_dir)
        print "  prefix: %s" % prefixes_by_flowcell[flowcell]
        print "  names:"
        for run, df in runs_by_date.iteritems(): 
            if df.iloc[0]["Flow Cell"] != flowcell:
                continue
            run_name = re.findall(r"run_(\d+)", run)[0]
            for f5_fname in f5_files_by_flowcell[flowcell]:
                try:
                    if "run_%s" % run_name in re.findall(r"(run_\d+_\d+.fast5)", f5_fname)[0]:
                        r = re.findall(r"(run_\d+_\d+.fast5)", f5_fname)[0]
                        print "    run%s: %s" % (run_name, r)
                except IndexError:
                    pass
        print "  run_splits:"
        formatted_coords = {}
        for run, df in runs_by_date.iteritems(): 
            if df.iloc[0]["Flow Cell"] != flowcell:
                continue
            formatted_coords[run] = [] 
            r = re.findall(r"run_(\d+)", run)
            print "    run%s:" % r[0]
            mod = 0
            for i, coords in enumerate(df.loc[:, ["start (sec)", "end (sec)"]].iterrows()):
                letter = alpha[i - mod]
                if np.isnan(coords[1][0]):
                    mod += 1
                    continue
                else:
                    start = int(coords[1][0])
                if np.isnan(coords[1][1]):
                    end = start + 100
                else:
                    end = int(coords[1][1])
                print "    - name: %s" % letter
                print "      start: %d" % start
                print "      end: %d" % end
                formatted_coords[run].append({"name": letter, "start": start, "end": end})
        print "segmentation_params:"
        print "  out_prefix: /disk1/pore_data/segmented/peptides/%s" % date
        print "  min_duration_obs:", min_duration_obs
        print "  signal_threshold:", signal_threshold
        print "  signal_priors:"
        print "    prior_open_pore_mean:", open_pore_mean
        print "    prior_open_pore_std:", open_pore_stdv


# ## Write to config file(s)

# In[17]:


for flowcell in flowcells:
    if f5_files_by_flowcell[flowcell]:
        with open(config_files_by_flowcell[flowcell], "w+") as f:
            f.write("fast5:\n")
            f.write("  dir: %s\n" % os.path.join(f5_base_dir, f5_dir))
            f.write("  prefix: %s\n" % prefixes_by_flowcell[flowcell])
            f.write("  names:\n")
            for run, df in runs_by_date.iteritems():
                if df.iloc[0]["Flow Cell"] != flowcell:
                    continue
                run_name = re.findall(r"run_(\d+)", run)[0]
                for f5_fname in f5_files_by_flowcell[flowcell]:
                    try:
                        if "run_%s" % run_name in re.findall(r"(run_\d+_\d+.fast5)", f5_fname)[0]:
                            r = re.findall(r"(run_\d+_\d+.fast5)", f5_fname)[0]
                            f.write("    run%s: %s\n" % (run_name, r))
                    except IndexError:
                        pass
            f.write("  run_splits:\n")
            formatted_coords = {}
            for run, df in runs_by_date.iteritems():
                if df.iloc[0]["Flow Cell"] != flowcell:
                    continue
                formatted_coords[run] = [] 
                r = re.findall(r"run_(\d+)", run)
                f.write("    run%s:\n" % r[0])
                mod = 0
                for i, coords in enumerate(df.loc[:, ["start (sec)", "end (sec)"]].iterrows()):
                    letter = alpha[i - mod]
                    if np.isnan(coords[1][0]):
                        mod += 1
                        continue
                    else:
                        start = int(coords[1][0])
                    if np.isnan(coords[1][1]):
                        end = start + 100
                    else:
                        end = int(coords[1][1])
                    f.write("    - name: %s\n" % letter)
                    f.write("      start: %d\n" % start)
                    f.write("      end: %d\n" % end)
                    formatted_coords[run].append({"name": letter, "start": start, "end": end})
            f.write("segmentation_params:\n")
            f.write("  out_prefix: /disk1/pore_data/segmented/peptides/%s\n" % date)
            f.write("  min_duration_obs: %d\n" % min_duration_obs)
            f.write("  signal_threshold: %f\n" % signal_threshold)
            f.write("  signal_priors:\n")
            f.write("    prior_open_pore_mean: %f\n" % open_pore_mean)
            f.write("    prior_open_pore_std: %f\n" % open_pore_stdv)


# # Generate ipython notebook(s)

# Separate ipython notebooks are generated for runs with separate flowcells.

# In[18]:


for flowcell in flowcells:
    if f5_files_by_flowcell[flowcell]:
        template_fname = "experiment_TEMPLATE.ipynb"
        notebook_fname = "experiments/experiment_%s_%s.ipynb" % (date, flowcell)
        with open(template_fname, "r") as template_nb:
            lines = template_nb.readlines()
            lines = "\n".join(lines)
            lines = lines.replace("INSERT_DATE", date)
            lines = lines.replace("INSERT_FLOWCELL", flowcell)
        with open(notebook_fname, "w+") as nb:
            nb.write(lines)


# In[ ]:




