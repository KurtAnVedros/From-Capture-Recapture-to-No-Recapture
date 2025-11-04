import numpy
import scipy.linalg


def DeduceVhiFilter(training_set, cutting_point):

    u, s, v_t = numpy.linalg.svd(training_set, full_matrices=False)
    v = numpy.transpose(v_t)
    new_v = v[:, 0:cutting_point]

    return new_v


def ApplyVhiFilter(dataset, vhi):

    denoised_dataset = numpy.dot(numpy.dot(dataset, vhi), numpy.transpose(vhi))

    return denoised_dataset


def DeduceSigmaFilter(training_set, cutting_point):

    u, s, v_transpose = numpy.linalg.svd(training_set, full_matrices=False)

    new_s = [value if index < cutting_point else 0 for index, value in enumerate(s)]
    s_clean = numpy.diag(new_s)

    return s_clean


def ApplySigmaFilter(dataset, sigma_clean):

    u, s, v_transpose = numpy.linalg.svd(dataset, full_matrices=False)
    denoised_dataset = numpy.dot(numpy.dot(u, sigma_clean), v_transpose)

    return denoised_dataset





#============Below this point older/depricated functions==================================





def ApplySigmaBasedSVD(dataset, cutting_point):

    u, s, v_transpose = numpy.linalg.svd(dataset, full_matrices=False)

    new_s = [value if index < cutting_point else 0 for index, value in enumerate(s)]
    s_matrix = numpy.diag(new_s)
    denoised_dataset = numpy.dot(numpy.dot(u, s_matrix), v_transpose)

    return denoised_dataset





def SigmaBasedSVDDenoiseStreamingOne(set, hankel_rows, cutting_point):


    denoised_set = numpy.zeros(set.shape)

    for index, signal in enumerate(set):
        hankel_signal = scipy.linalg.hankel(signal[0:hankel_rows], signal[hankel_rows-1:])
        u, s, v = numpy.linalg.svd(hankel_signal, full_matrices=False)
        new_s = [value if i < cutting_point else 0 for i, value in enumerate(s)]
        s_matrix = numpy.diag(new_s)
        hankel_denoised_signal = numpy.dot(numpy.dot(u, s_matrix), v)
        hankel_denoised_signal_a = hankel_denoised_signal[:, 0]
        hankel_denoised_signal_b = hankel_denoised_signal[-1, 1:]
        denoised_signal = numpy.concatenate((hankel_denoised_signal_a, hankel_denoised_signal_b))
        denoised_set[index] = denoised_signal

    return denoised_set





def SigmaBasedSVDDenoiseStreaming(training_set, testing_set, cutting_point):

    u, s, v = numpy.linalg.svd(training_set, full_matrices=False)

    new_s = [value if index < cutting_point else 0 for index, value in enumerate(s)]
    s_matrix = numpy.diag(new_s)
    denoised_training_set = numpy.dot(numpy.dot(u, s_matrix), v)




    denoised_test_set = numpy.zeros(testing_set.shape)
    no_rows = denoised_test_set.shape[0]
    for index, signal in enumerate(testing_set):
        hankel_signal = scipy.linalg.hankel(signal[0:no_rows], signal[no_rows-1:])
        u, s, v = numpy.linalg.svd(hankel_signal, full_matrices=False)
        new_s = [value if i < cutting_point else 0 for i, value in enumerate(s)]
        s_matrix = numpy.diag(new_s)
        hankel_denoised_signal = numpy.dot(numpy.dot(u, s_matrix), v)
        hankel_denoised_signal_a = hankel_denoised_signal[:, 0]
        hankel_denoised_signal_b = hankel_denoised_signal[-1, 1:]
        denoised_signal = numpy.concatenate((hankel_denoised_signal_a, hankel_denoised_signal_b))
        denoised_test_set[index] = denoised_signal

    return denoised_training_set, denoised_test_set





def SigmaBasedSVDDenoiseDeduceSigma(training_set, cutting_point):

    u, s, v_transpose = numpy.linalg.svd(training_set, full_matrices=False)

    new_s = [value if index < cutting_point else 0 for index, value in enumerate(s)]
    s_clean = numpy.diag(new_s)

    return s_clean


def SigmaBasedSVDDenoiseApplyFilter(dataset, sigma_clean):

    u, s, v_transpose = numpy.linalg.svd(dataset, full_matrices=False)
    denoised_dataset = numpy.dot(numpy.dot(u, sigma_clean), v_transpose)

    return denoised_dataset








def VhiBasedSVDDenoise(dataset, vhi):


    denoised_dataset = numpy.dot(dataset, vhi)

    return denoised_dataset


def VhiBasedSVDDenoise2(dataset, vhi):


    denoised_dataset = numpy.dot(numpy.dot(dataset, vhi), numpy.transpose(vhi))

    return denoised_dataset





def VhiBasedSVDDenoiseStreamingNoHankel(dataset, vhi):


    # denoise a test set, record after record (streaming)
    rows = dataset.shape[0]
    columns = vhi.shape[1]

    denoised_dataset = numpy.zeros((rows, columns))
    for index, signal in enumerate(dataset):
        denoised_signal = numpy.dot(signal, vhi)
        denoised_dataset[index] = denoised_signal

    return denoised_dataset

