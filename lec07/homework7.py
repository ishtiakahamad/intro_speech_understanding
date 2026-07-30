import numpy as np

def major_chord(f, Fs):
    
    t = np.arange(int(0.5 * Fs)) / Fs
    f1 = f
    f2 = f * 2**(4/12)
    f3 = f * 2**(7/12)
    x = np.cos(2*np.pi*f1*t) + np.cos(2*np.pi*f2*t) + np.cos(2*np.pi*f3*t)
    
    return x


def dft_matrix(N):
    
    k = np.arange(N).reshape(-1, 1)
    n = np.arange(N).reshape(1, -1)
    W = np.exp(-1j * 2 * np.pi * k * n / N)
    
    return W.astype('complex')


def spectral_analysis(x, Fs):

    
    N = len(x)
    W = dft_matrix(N)
    X = W @ x
    mag = np.abs(X)

    half = N // 2
    mag_half = mag[:half]

    idx = np.argsort(mag_half)[::-1][:3]
    idx = np.sort(idx)

    freqs = idx * Fs / N
    f1, f2, f3 = freqs[0], freqs[1], freqs[2]
    
    return f1, f2, f3
