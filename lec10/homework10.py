import numpy as np
import torch, torch.nn


def get_features(waveform, Fs):
   
    waveform = np.asarray(waveform, dtype=float)

    alpha = 0.95
    
    emphasized = np.append(waveform[0], waveform[1:] - alpha * waveform[:-1])

    frame_len = int(round(0.004 * Fs))
    frame_step = int(round(0.002 * Fs))
    
    nframes_feat = 1 + (len(emphasized) - frame_len) // frame_step

    window = np.hamming(frame_len)
    half = frame_len // 2  #

    features = np.zeros((nframes_feat, half))
    
    for i in range(nframes_feat):
        start = i * frame_step
        frame = emphasized[start:start + frame_len] * window
        spec = np.abs(np.fft.fft(frame))
        features[i, :] = spec[:half]

    vad_win = int(round(0.025 * Fs))
    vad_step = int(round(0.010 * Fs))
    nframes_vad = 1 + (len(waveform) - vad_win) // vad_step

    log_energy = np.zeros(nframes_vad)
    
    for i in range(nframes_vad):
        start = i * vad_step
        frame = waveform[start:start + vad_win]
        e = np.sum(frame ** 2)
        log_energy[i] = 10 * np.log10(e + 1e-12)

    threshold = np.max(log_energy) - 30
    is_speech = log_energy > threshold

    vad_labels = np.zeros(nframes_vad, dtype=int)
    current_label = 0
    prev = False
    
    for i in range(nframes_vad):
        if is_speech[i]:
            if not prev:
                current_label += 1
            vad_labels[i] = current_label
        prev = is_speech[i]

    labels = np.repeat(vad_labels, 5)

    n = min(len(features), len(labels))
    features = features[:n]
    labels = labels[:n]

    return features, labels


def train_neuralnet(features, labels, iterations):
   
    nfeats = features.shape[1]
    nlabels = int(np.max(labels)) + 1

    model = torch.nn.Sequential(
        torch.nn.LayerNorm(nfeats),
        torch.nn.Linear(nfeats, nlabels)
    )

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    lossvalues = np.zeros(iterations)
    
    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        lossvalues[i] = loss.item()

    return model, lossvalues


def test_neuralnet(model, features):
    
    X = torch.tensor(features, dtype=torch.float32)
    outputs = model(X)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.detach().numpy()
