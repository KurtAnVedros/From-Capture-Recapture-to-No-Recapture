import numpy
from scipy.fftpack import fft
from scipy.fftpack import ifft
import math
from itertools import combinations



#let us assume that both Q and T are lists corresponding to a single signal
def slidingDotProduct(Q, T):

    #n <- Length(T), m <- Length(Q)
    n = T.shape[0]
    m = Q.shape[0]
    # Ta <- Append T with n zeros
    Ta = numpy.append(T, numpy.zeros(m))
    # Qr <- Reverse(Q)
    Qr = Q[::-1]
    # Qra <- Append Qr with 2n-m zeros
    Qra = numpy.append(Qr, numpy.zeros(n))
    #Qra = Qr
    # Qraf <- FFT(Qra)
    Qraf = fft(Qra)
    # Taf <- FFT(Ta)
    Taf = fft(Ta)
    # QT <- InverseFFT(ElementwiseMultiplication(Qraf, Taf))

    #mult = [a * b for a, b in zip(Qraf, Taf)]
    mult = numpy.multiply(Qraf, Taf)
    #QT = ifft(mult)
    QT = ifft(mult).real

    return QT


def calculateDistanceProfile(Q, T, QT, uT, sT, index):

    m = Q.shape[0]
    n = T.shape[0]
    #D = numpy.zeros(m)



    #for i in range(0, m):
    #    D[i] = math.sqrt((2 * m) * (1 - (QT[i] - m * uQ * Mt[i]) / (m * sQ * St[i])))
    test0 = QT[m-1: n]
    test1 = test0 - m * uT * uT[index]
    test2 = sT * sT[index]
    #print(test2[16125:])
    test3 = test1/test2
    D = 2 * (m - test3)
    #print(D[16125:16128])
    D = numpy.abs(D)
    D = numpy.sqrt(D)

    return D

def computeMeanStd(T, m):

    n = T.shape[0]
    data_cum_sum = numpy.cumsum(T)
    data2_cum_sum = numpy.cumsum(numpy.multiply(T, T))
    test1 = data2_cum_sum[m-1:n+m+1]
    test2 = numpy.insert(data2_cum_sum[0:(n - m)], 0, 0)
    data2_sum = numpy.subtract(test1, test2)
    #data2_sum = numpy.subtract(data2_cum_sum[m:n+m+1], numpy.insert(data2_cum_sum[0:(n - m)], 0, 0))
    test3 = data_cum_sum[(m - 1):n]
    test4 = numpy.insert(data_cum_sum[0:(n - m)], 0, 0)
    data_sum = test3 - test4
    #data_sum = numpy.subtract(data_cum_sum[(m - 1):n], numpy.insert(data_cum_sum[0:(n - m)], 0, 0))
    #data_Mu = [n / m for n in data_sum]
    data_Mu = numpy.divide(data_sum, m)
    #data2_Sig = ([n / m for n in data2_sum]) - ([n ** 2 for n in data_Mu])
    data2_Sig = numpy.subtract(numpy.divide(data2_sum, m), numpy.multiply(data_Mu, data_Mu))
    #data_Sig = [math.sqrt(n) for n in data2_Sig]
    data_Sig = numpy.sqrt(data2_Sig)

    return data_Mu, data_Sig

def calculateDiagonalDist(Q, T, QT, uT, sT, index, m):

    n = T.shape[0]
    p = n - m + 1

    a = numpy.repeat(1, p - index).reshape(p - index, 1)
    b = T[index:(index + m)].reshape(1, m)
    #xxx = numpy.tile(b, p - index).reshape(p - index, m)
    c = T[0:m].reshape(m, 1)
    xxx = a.dot(b).dot(c)
    #yyy = a * b * c




    #xTerm = (numpy.repeat(1, n - index + 1)) * T[(index + m - 1):index+1:-1] * T[1:m]
    xTerm = a.dot(b).dot(c)
    mTerm = T[index:p - 1] * T[0:(p - index - 1)]
    aTerm = T[index + m:] * T[m:(n - index)]

    if p is not index:
        xTerm[1:] = xTerm[1:] - numpy.cumsum(mTerm).reshape(mTerm.shape[0], 1) + numpy.cumsum(aTerm).reshape(mTerm.shape[0], 1)


    a = xTerm
    b = m * uT[index:]
    c = uT[0:p - index]
    d = m * sT[index:]
    e = sT[0:p - index]

    g = b.reshape(b.shape[0], 1) * c.reshape(c.shape[0], 1)
    h = a.reshape(a.shape[0], 1) - g.reshape(g.shape[0], 1)
    j = d.reshape(d.shape[0], 1) * e.reshape(e.shape[0], 1)
    D = h / j


    #D = (a - g) / (d * e)
    #D = (xTerm - m * uT[index:] * uT[0:p - index + 1]) / (m * sT[index:] * sT[0:p - index + 1])

    D = 2 * m * (1 - D)

    D = numpy.abs(D)
    D = numpy.sqrt(D)


    return D


def MASS(Q, T, index):

    #QT <- SlidingDotProducts(Q, T)
    QT = slidingDotProduct(Q, T)
    #uQ, sQ, uT, sT<- computeMeanStd(Q, T)
    uT, sT = computeMeanStd(T, 256)
    #uQ, sQ = computeMeanStd(Q, 256)


    #D <- CalculateDistanceProfile(Q, T, QT, uQ, sQ, Mt, St)
    D = calculateDiagonalDist(Q, T, QT, uT, sT, index, 256)
    return D

#INPUT: (a) T_A vector in numpy array (for efficiency purposes) representing the base signal which will be examined
#       (b) T_B vector as numpy array representing the test signal from which the queries will be taken
#       (c) m is the size of the query
def STAMP(T_A, T_B, m):

    #nB <- len(Tb)
    #n_B = len(T_B) for vectors
    n_B = T_B.shape[0]
    n_A = T_A.shape[0]
    #Pab <- infs, Iab <- zeros, idxes <- 1:nB-m+1
    P_AB = numpy.repeat(math.inf, n_B - m + 1).reshape(n_B - m + 1, 1)
    I_AB = numpy.zeros(n_B - m + 1)
    idxes = list(range(int(m/2), n_A - m + 1))
    p = n_A - m + 1


    #B = [list(x) for x in combinations(T_B, m)]
    # select the query by choosing the next starting index upon the signal. The max index is given as: |T|-m+1
    for idx in idxes:

    #   D <- MASS(B[idx], Ta)
        B = T_B[idx:(idx + m)]
        D = MASS(B, T_A, idx)
    #   Pab, Iab <- ElementWiseMin(Pab, Iab, D, idx)
        #make nan as inf

        #D[numpy.isnan(D)] = numpy.inf

        #excluding_zone_start = int(max(0, idx - m/2 + 1))
        #excluding_zone_end = int(min(n_A - 1, idx + m/2 + 1))
        #D[excluding_zone_start:excluding_zone_end] = numpy.inf

        P_AB = numpy.minimum(P_AB, D)

    return P_AB, I_AB









