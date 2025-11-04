import matplotlib
import numpy


def plotTimeIntervals(values, labels):

    matplotlib.pyplot.rcdefaults()
    fig, ax = matplotlib.pyplot.subplots()

    y_pos = numpy.arange(len(labels))


    #ax.barh(y_pos, performance, xerr=error, align='center', color='green', ecolor='black')
    ax.barh(y_pos, values, align="center", color="green")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Procedures")
    ax.set_title("Execution Time")

    matplotlib.pyplot.show()
