#!/usr/bin/env python
# coding: utf-8

# Contains code to check and convert csv data on captured EM signals... I.E. the following.
# - Convert EM signals of csv to numpy for speed loading and manipulation.
# - Remove start and end common code as this is not appart of the main programs.
# - Test to check if all instructions in the code is accounted for in the signal.

# In[ ]:


from matplotlib import pyplot as plt
import random
import tensorflow as tf
import numpy as np
import timeit
import scipy.io as sio
import csv
import os
import math 
import random as rand
from numpy import genfromtxt
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal

import warnings
from scipy.signal import savgol_filter
from scipy.spatial import distance


import sys
sys.path.append('../../Imports/')

from Modules.Tools import MapTool
from Modules.Tools import csv2Numpy
from Modules.Tools import BoxPlot
from Modules.Tools import peakCorrelation as pC


# In[ ]:


'''
Method will obtain only the major peaks of the signal in the range of the peaks provided.

Signals: signals to extract the major peaks.
Start_Peak: Starting peak to extract from each signal.
End_Peak: Final peak to exatract from each signal
'''
def ObtainMajorPeaks(Signals, Start_Peak, End_Peak, pad, evens=True):
    ## extract the peaks only
    Baseline_peaks = MapTool.getPeaksDataset(Signals, pad)
    ## remove the end peak
    Baseline_peaks = Baseline_peaks[:,:Baseline_peaks.shape[1]-1]

    ## obtain only the major peaks removing the inbetween low peak.
    Baseline = []
    for n in range(Baseline_peaks.shape[0]):
        major_peaks = []
        for q in range(Baseline_peaks[n].shape[0]):
            if evens:
                if q%2 == 0:
                    major_peaks.append(Baseline_peaks[n][q])
            else:
                if q%2 != 0:
                    major_peaks.append(Baseline_peaks[n][q])
        Baseline.append(major_peaks)

    Baseline = np.array(Baseline, dtype=object)
    
    return Baseline


# In[ ]:


'''
Method takes the csv folders in a file location and converts the me into numpy signals. 
    Saves and graphs the full signal, the segments containing the main program (w/o start and end), and the major peaks of the segements.
    
Input:
    folder: location of the csv files to extract from.
    folder_name: folder_name of which to save the data as.
    name: name of which to save the data as.
    main_code: name of the instance to save as.
    add_main_code: names to add to save the folder if need be.
    test_loc: id location to graph the signal samples.
    start_peak: start of the main program segment, should be after the start instructions.
    end_peak: end of the main program segment, should be before the end instructions.
    start_code: code that is commonly started for the program but is not part of the main program under consideration.
    end_code: code that is commonly at the end for the program but is not part of the main program under consideration.
'''
def convert2numpy(folder, folder_name, name, save_graph_folder, save_np_folder, main_code, add_main_code="",
                  test_loc=0, 
                  start_peak = 20, end_peak=24, start_code=[], 
                  end_code=[], y_low=-2.5, y_high=2.5, pad=10, amount=3000, majorPeaks=True, skip_start_peaks=0, evens=False):
    
    actual_folder = folder + "/"
    
    # Obtain the full signal
    ## Cut to the length of the B channel.
    signal_Full = csv2Numpy.csv2numpyCount(actual_folder, 2, toCut = True, thresh= 1, direction = 1, amount = amount)
    
    ## normalize (Note: This normalizes by the common instructions excluding the first that is influence by B channel change.)
    signal_Full = pC.normalizeSec(signal_Full, 5, 23, pad, convert="average")
    
    ## Graph Full Signal
    if add_main_code == "":
        code = start_code + main_code + end_code
    else:
        code = start_code + add_main_code + end_code
    
    Signal = signal_Full[test_loc]

    time_Signal = np.arange(Signal.shape[0])
    plt.plot(time_Signal, Signal)
    MapTool.mapCodeSignal(Signal, code, pad)
    plt.axvspan(MapTool.getPeaksLoc(Signal, pad, start_peak,getHigh = False), MapTool.getPeaksLoc(Signal, pad, end_peak), color='orange', alpha=0.5)
    #plt.title("Manual Mapping for Full Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(axis='y')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              fancybox=True, shadow=True, ncol=6)
    plt.xlim([0,len(Signal)])
    plt.ylim(y_low,y_high)
    folder = save_graph_folder + "full/" + folder_name + "/"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    save_plt_folder = save_graph_folder + "full/" + folder_name + "/" + name + "_full.png"
    plt.savefig(save_plt_folder, bbox_inches='tight')
    plt.show()
    plt.close()
    
    folder = save_np_folder + "full" + "/" + folder_name + "/"
    if not os.path.isdir(folder):
        os.makedirs (folder)
    np.save(save_np_folder + "full" + "/" + folder_name + "/" + name + "_full.npy", signal_Full)
    
    signal_Seg = pC.getSeqSignals(signal_Full, start_peak, end_peak, pad)
    
    code = main_code
    
    Signal = signal_Seg[test_loc]

    time_Signal = np.arange(Signal.shape[0])
    plt.plot(time_Signal, Signal)
    MapTool.mapCodeSignal(Signal, code, pad, skip_start_peaks=skip_start_peaks)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(axis='y')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              fancybox=True, shadow=True, ncol=6)
    plt.xlim([0,len(Signal)])
    plt.ylim(y_low,y_high)
    
    folder = save_graph_folder + "segment" + "/" + folder_name + "/"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    save_plt_folder = save_graph_folder + "segment/" + folder_name + "/" + name + "_segment.png"    
    plt.savefig(save_plt_folder, bbox_inches='tight')
    plt.show()
    plt.close()
    
    folder = save_np_folder + "segment" + "/" + folder_name + "/"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    np.save(save_np_folder + "segment" + "/" + folder_name + "/" + name + "_segment.npy", signal_Seg)
    
    
    if majorPeaks == True:
        signal_Major_Peaks = ObtainMajorPeaks(signal_Seg, start_peak, end_peak, pad, evens=evens)

        Signal = signal_Major_Peaks[test_loc]

        time_Signal = np.arange(Signal.shape[0])
        plt.plot(time_Signal, Signal)
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(axis='y')

        plt.xlim([0,len(Signal)])
        folder = save_graph_folder + "Major_Peaks/" + folder_name + "/"
        if not os.path.isdir(folder):
            os.makedirs(folder)
        save_plt_file = save_graph_folder + "Major_Peaks/" + folder_name + "/" + name + "_major_peaks.png"    
        plt.savefig(save_plt_file, bbox_inches='tight')
        plt.show()
        plt.close()

        folder = save_np_folder + "Major_Peaks/" + folder_name + "/"
        if not os.path.isdir(folder):
            os.makedirs(folder)
        np.save(save_np_folder + "Major_Peaks" + "/" + folder_name + "/" + name + "_major_peaks.npy", signal_Major_Peaks)


# In[ ]:


'''
Method takes the csv folders in a file location and converts them into numpy signals and graphs for testing. 
 
 Input:
    folder: location of the csv files to extract from.
    folder_name: folder_name of which to save the data as.
    main_code: code of the main program.
    test_loc: id location to graph the signal samples.
    start_peak: start of the main program segment, should be after the start instructions.
    end_peak: end of the main program segment, should be before the end instructions.
    start_code: code that is commonly started for the program but is not part of the main program under consideration.
    end_code: code that is commonly at the end for the program but is not part of the main program under consideration.
'''
def test(folder, folder_name, name, save_folder, main_code, add_main_code="",
                  test_loc=0, 
                  start_peak = 20, end_peak=24, start_code=[], 
                  end_code=[], y_low=-2.5, y_high=2.5, pad=10):
    
    actual_folder = folder + "/"
    
    # Obtain the full signal
    ## Cut to the length of the B channel.
    signal_Full = csv2Numpy.csv2numpyCount(actual_folder, 2, toCut = True, thresh=1, direction = 1, amount = 10)
    
    ## normalize (Note: This normalizes by the common instructions excluding the first that is influence by B channel change.)
    signal_Full = pC.normalizeSec(signal_Full, 5, 23, pad, convert="average")
    
    ## Graph Full Signal
    if add_main_code == "":
        code = start_code + main_code + end_code
    else:
        code = start_code + add_main_code + end_code
    
    Signal = signal_Full[test_loc]

    time_Signal = np.arange(Signal.shape[0])
    plt.plot(time_Signal, Signal)
    MapTool.mapCodeSignal(Signal, code, pad)
    plt.axvspan(MapTool.getPeaksLoc(Signal, pad, start_peak), MapTool.getPeaksLoc(Signal, pad, end_peak), color='orange', alpha=0.5)
    #plt.title("Manual Mapping for Full Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(axis='y')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              fancybox=True, shadow=True, ncol=6)
    plt.xlim([0,len(Signal)])
    plt.ylim(y_low,y_high)
    save_plt_folder = save_folder + "full/" + folder_name + "/"
    if not os.path.isdir(save_plt_folder):
        os.makedirs (save_plt_folder)
        
    save_plt_file = save_folder + "full/" + folder_name + "/" + name + "_full.png"
    plt.savefig(save_plt_file, bbox_inches='tight')
    plt.show()
    
    Signal = signal_Full[test_loc]

    time_Signal = np.arange(Signal.shape[0])
    plt.plot(time_Signal, Signal)
    MapTool.mapCodeSignal(Signal, code, pad)
    plt.axvspan(MapTool.getPeaksLoc(Signal, pad, start_peak), MapTool.getPeaksLoc(Signal, pad, end_peak), color='orange', alpha=0.5)
    #plt.title("Manual Mapping for Full Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(axis='y')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              fancybox=True, shadow=True, ncol=6)
    plt.xlim([len(Signal)-300,len(Signal)])
    plt.ylim(y_low,y_high)
    plt.show()
    plt.close()

