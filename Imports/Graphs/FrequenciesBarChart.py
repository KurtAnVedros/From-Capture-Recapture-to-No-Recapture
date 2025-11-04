import matplotlib.pyplot
import numpy
from IO import Writer



def getAmplitudeFrequencyDomain(sampling_rate, total_samples, fft_values):


    n = total_samples  # length of the analog signal
    k = numpy.arange(n)
    T = n / sampling_rate
    xx2 = k / T  # two sides frequency range
    #xx2 = xx2[range(n // 2)]  # one side frequency range
    yy2 = fft_values / n
    #yy2 = yy2[range(n // 2)]

    return xx2, yy2

def filterNoise(clean_signals, noisy_signals):

    mean_freq_clean_signal = numpy.median(clean_signals, axis=0)

    filtered_signals = noisy_signals
    for i, signal in enumerate(noisy_signals):
        for j, sample in enumerate(signal):
            if abs(sample) > abs(mean_freq_clean_signal[j]):
                filtered_signals[i][j] = mean_freq_clean_signal[j]

    return filtered_signals

def plotFrequencies2(xx, clean, noisy, filtered, output_file):

    fig, ax = matplotlib.pyplot.subplots(3, 1)

    ax[0].plot(xx, abs(clean), 'g')
    matplotlib.pyplot.ylim(0, 20)
    ax[0].set_xlabel("Freq (Hz)")
    ax[0].set_ylabel("|Y(freq)|")

    ax[1].plot(xx, abs(noisy), 'r')  # plotting the spectrum
    matplotlib.pyplot.ylim(0, 20)
    ax[1].set_xlabel("Freq (Hz)")
    ax[1].set_ylabel("|Y(freq)|")

    ax[2].plot(xx, abs(filtered), 'y')  # plotting the spectrum
    matplotlib.pyplot.ylim(0, 20)
    ax[2].set_xlabel("Freq (Hz)")
    ax[2].set_ylabel("|Y(freq)|")

    # plot_url = matplotlib.pyplot.plot_mpl(fig, filename='/Users/Hellboy/Desktop/mpl-basic-fft.png')

    Writer.savePlot(output_file)


def plotFrequencies1(xx, yy, output_file):


    fig = matplotlib.pyplot.figure()
    ax = matplotlib.pyplot.subplot(111)
    matplotlib.pyplot.ylim(0, 20)
    ax.plot(xx, yy, label='yy')
    matplotlib.pyplot.title('Legend')
    ax.legend()
    fig.savefig(output_file)



def plotFrequencies(original_values, fft_values, frequency, sampling_rate, output_file):

    Fs = sampling_rate
    Ts = 1.0 / Fs
    #xx1 = numpy.arange(0, 1, Ts)  # xx' time
    #yy1 = original_values
    #yy1 = numpy.sin(2 * numpy.pi * 5 * xx1) # yy' amplitude

    n = len(original_values) # length of the analog signal
    k = numpy.arange(n)
    T = n / sampling_rate
    xx2 = k / T  # two sides frequency range
    #xx2 = xx2[range(n // 2)]  # one side frequency range
    yy2 = fft_values / n
    #yy2 = yy2[range(n // 2)]




    fig, ax = matplotlib.pyplot.subplots(2, 1)

    #ax[0].plot(xx1, yy1)
    #ax[0].set_xlabel('Time')
    #ax[0].set_ylabel('Amplitude')

    ax[1].plot(xx2, abs(yy2), 'r')  # plotting the spectrum
    #matplotlib.pyplot.ylim(0, 20)
    ax[1].set_xlabel("Freq (Hz)")
    ax[1].set_ylabel("|Y(freq)|")

    #plot_url = matplotlib.pyplot.plot_mpl(fig, filename='/Users/Hellboy/Desktop/mpl-basic-fft.png')

    Writer.savePlot(output_file)

    return xx2, yy2



def plotFrequencies2(x, y, y_limit_min, y_limit_max, output_path):


    matplotlib.pyplot.plot(x, y)
    axes = matplotlib.pyplot.gca()

    axes.set_ylim([y_limit_min, y_limit_max])

    matplotlib.pyplot.savefig(output_path)
    matplotlib.pyplot.close()