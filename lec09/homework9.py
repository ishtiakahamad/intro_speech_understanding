import numpy as np

def VAD(waveform, Fs):
    
    frame_length = int(0.025 * Fs)
    step = int(0.010 * Fs)

    frames = waveform_to_frames(waveform, frame_length, step)
    energy = np.sum(frames**2, axis=1)
    
    threshold = 0.1 * np.amax(energy)
    mask = energy > threshold

    segments = []
    n = len(mask)
    i = 0
    
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
          
            start = i * step
            end = (j - 1) * step + frame_length
            segments.append(waveform[start:end])
            i = j
        else:
            i += 1

    return segments


def segments_to_models(segments, Fs):
 
    frame_length = int(0.004 * Fs)
    step = int(0.002 * Fs)

    models = []
    for segment in segments:
      
        preemph = np.append(segment[0], segment[1:] - 0.95 * segment[:-1])

        frames = waveform_to_frames(preemph, frame_length, step)
        mstft = frames_to_mstft(frames)
        spectrogram = mstft_to_spectrogram(mstft)

        half = frame_length // 2
        low_freq = spectrogram[:, :half]

        model = np.mean(low_freq, axis=0)
        models.append(model)

    return models


def recognize_speech(testspeech, Fs, models, labels):
   
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)

    Y = len(models)
    K = len(test_models)
    sims = np.zeros((Y, K))

    for y in range(Y):
        a = models[y]
        for k in range(K):
            b = test_models[k]
            sims[y, k] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    test_outputs = []
    for k in range(K):
        best_y = np.argmax(sims[:, k])
        test_outputs.append(labels[best_y])

    return sims, test_outputs
