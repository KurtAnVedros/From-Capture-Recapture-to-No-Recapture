import numpy as np
from scipy.stats import zscore
from scipy.fftpack import irfft, rfft


def movstd(array, back_win, front_win):
    curr_index = 0
    n = len(array)
    std_list = []
    while (curr_index < n):
        print(str(curr_index)+" "+str(n))
        if (curr_index - front_win < 0):
            front = 0
        else:
            front = curr_index - front_win
        if ((curr_index + back_win + 1) > n):
            back = n
        else:
            back = curr_index + back_win + 1
        window = array[front:back]
        std_list.append(np.std(window))
        curr_index += 1
    return std_list



def executeNovel(flattened_T,query_set,top_n=10):
    strangeness_list=[]
    for signal in query_set:
        score=np.sum(novel_metric(flattened_T,signal)[:top_n])
        strangeness_list.append(score)
    return strangeness_list


def novel_metric(T,Q):
    #repeat for each signal of T vs T, each signal of test set vs T
    m=len(Q)
    n=len(T)
    print("move before")
    std_list = movstd(T, 1, m)
    print("mov over")
    d = np.zeros((n))
    Q_norm = zscore(Q)
    Q_rev = np.flip(Q_norm)
    padding = np.zeros((n - m))
    print("got here")
    Q_rev = np.concatenate((Q_rev, padding))
    p = irfft(rfft(T) * rfft(Q_rev))
    print("also")
    d = np.sqrt(np.divide((2 * (m - p)), std_list))
    print("done")
    return np.sort(d)
