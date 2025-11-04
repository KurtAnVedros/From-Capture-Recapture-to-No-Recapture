import matplotlib.pyplot as plt



def plotTimeSeries(x, y, title, output_path):

    plt.plot(x, y)
    axes = plt.gca()
    axes.set_title(title)
    axes.set_xlabel('Sample Index')
    axes.set_ylabel('Amplitude')
    plt.savefig(output_path)
    plt.close()

