import numpy as np
from scipy.special import erfc


def is_array(obj):
    if isinstance(obj, (dict, str)): return False
    return hasattr(obj, "__len__")


def confsmooth(y, noise_level, confidence=0.999, overlap_fraction=0.5, deg=2):
    """
    Smooth experimental spectrum, keeping peak intensity

    :param y: function values
    :param noise_level: scalar or vector of the same size as y, containing standard deviations of the noise
    :param confidence: errors with probability < 1-confidence are treated as signal and are not smoothed
    :param overlap_fraction: how many percent of points of interval to use for overlap
    :param deg: degree of polynomial
    :returns: smoothed y
    """
    n = len(y)
    if not is_array(noise_level): noise_level = np.ones(n)*noise_level

    def approx_error(y):
        x_i = np.arange(len(y))  # we do not take x into account!
        p = np.polyfit(x_i, y, deg)
        pred = np.polyval(p, x_i)
        return pred, y-pred

    def is_noise_possible_ext(error, sigma):
        ind = sigma>0
        error = error[ind]
        if len(error) == 0:
            if np.any(error) > 1e-6: return False
            else: return True
        if np.all(error == 0): return True
        sigma = sigma[ind]
        noise_prob = erfc(np.abs(error)/sigma/np.sqrt(2))
        if np.any(noise_prob < 1-confidence): return False
        for j in range(2,len(error)//2):
            error1 = np.convolve(error,np.ones(j)/j, 'valid')
            sigma1 = sigma[j//2:j//2+len(error1)]/np.sqrt(j)
            noise_prob = erfc(np.abs(error1)/sigma1/np.sqrt(2))
            if np.any(noise_prob < 1-confidence): return False
        return True

    def is_noise_possible(i0, size):
        error = approx_error(y[i0:i0 + size])[1]
        return is_noise_possible_ext(error, noise_level[i0:i0+size])

    def detect_segment_size(i0, old_size):
        sz = old_size
        is_ns_possible = is_noise_possible(i0, sz)
        if is_ns_possible:
            while is_ns_possible:
                sz = sz + 1
                if i0+sz > len(y): return sz-1
                is_ns_possible = is_noise_possible(i0, sz)
            return sz-1
        else:
            while not is_ns_possible:
                sz = sz - 1
                assert sz >= 2
                is_ns_possible = is_noise_possible(i0, sz)
            return sz

    i0 = 0
    old_size = deg+1
    result = np.zeros(n)
    osz = 0
    while True:
        new_size = detect_segment_size(i0, old_size)
        # print(f'i0 = {i0} new_size = {new_size} old_size = {old_size} osz = {osz}')
        pred, error = approx_error(y[i0:i0 + new_size])
        if i0 == 0:
            result[:new_size] = pred
            osz = int(np.round(new_size*overlap_fraction))
            if osz == 0: osz = 1
        else:
            # overlap with old
            corrected_osz = min(osz, new_size)
            if corrected_osz >= 3:
                alpha = np.linspace(0,1,corrected_osz)
            else:
                alpha = np.ones(corrected_osz)/2
            result[i0:i0+corrected_osz] = (1-alpha)*result[i0:i0+corrected_osz] + alpha*pred[:corrected_osz]
            # middle
            result[i0+corrected_osz:i0+new_size] = pred[corrected_osz:]
            #overlap with new
            osz = int(np.round(new_size*overlap_fraction))
            if osz == 0: osz = 1
        if i0+new_size >= len(y): break
        i0 = i0 + new_size - osz
        old_size = new_size

    return result


