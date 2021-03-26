import matplotlib.pyplot as plt
import librosa
import numpy as np

def coinvolutions(signal):
    matrix = np.zeros((len(taus), n_windows))
    max_tau = np.zeros(n_windows)
    for t in range(n_windows):
        coinvolutions = []
        for tau in taus:
            sample_0 = signal[coinv_window*t : coinv_window*(t+1)]
            sample_1 = signal[coinv_window*t+tau : coinv_window*(t+1)+tau]
            coinvolutions.append(np.dot(sample_0, sample_1))
        matrix[:,t] = coinvolutions
        max_tau[t] = np.argmax(coinvolutions)
    return max_tau.astype(int)

def parabolic_interpolation(matrix,winners):
    """
    Applies parabolic interpolation onto the candidate period estimates using
    3 points corresponding to the estimate and it's adjacent values
    
    #Arguments
        matrix: An MxK array with the scores of all the M taus over the K windows\n
        winners: Array with the current winners estimates\n
    
    #Returns
        interpolated_winners: Array containing the interpolated period estimates
        for each sample.\n
    """
    new_winners = np.zeros_like(winners)
    for t,winner in enumerate(winners):
        x_points = np.arange(winner-1,winner+2).astype(int)
        y_points = matrix[x_points,t]
        coefficients = np.polyfit(x_points, y_points, 2)
        derivative = np.poly1d(coefficients).deriv()
        a = coefficients[0]
        b = coefficients[1]
        candidate = -b/(2*a+np.finfo(a).tiny)
        if 0 < candidate and candidate < len(taus):
            new_winners[t] = candidate
        else:
            new_winners[t] = winner
    return new_winners

def differences(signal, normalized = True, threshold = True, par_interpolation = True, local_estimate = True):
    matrix = np.zeros((len(taus), n_windows))
    winners = np.zeros(n_windows)
    for t in range(n_windows):
        d_array = []
        for j,tau in enumerate(taus):
            sample_0 = signal[coinv_window*t:coinv_window*(t+1)]
            sample_1 = signal[coinv_window*t+tau:coinv_window*(t+1)+tau]
            e_sample_0 = np.dot(sample_0, sample_0)
            e_sample_1 = np.dot(sample_1, sample_1)
            cross = np.dot(sample_0, sample_1)
            d = e_sample_0 + e_sample_1 - 2*cross
            if threshold and d < 0.1 and winners[t] == 0:
                winners[t] = j
            if(normalized and len(d_array) > 0):
                d_array.append(d/np.mean(d_array))
            else:
                d_array.append(d)
        matrix[:,t] = d_array
        if winners[t] == 0:
            winners[t] = np.argmin(d_array)

    if par_interpolation:
        winners = parabolic_interpolation(matrix,winners)

    if local_estimate:
        for i, winner in enumerate(winners):
            submatrix = matrix[int(winner*4//5):int(winner*5//4)+1,max(i-1,0):min(i+2,len(winners))]
            winner = winner*4//5 + np.argwhere(submatrix == np.min(submatrix))[0][0]       
    
    return winners

def step1(signal):
    """
    coinvolutions
    """
    winners = coinvolutions(signal)
    for i in range(len(winners)):
        winners[i] = sr/taus[int(winners[i])]
    return winners
def step2(signal):
    """
    differences
    """
    winners = differences(signal, normalized=False, threshold=False, par_interpolation = False, local_estimate = False)
    for i in range(len(winners)):
        winners[i] = sr/taus[int(winners[i])]
    return winners
def step3(signal):
    """
    cumulative mean normalized differences
    """
    winners = differences(signal, normalized=True, threshold=False, par_interpolation = False, local_estimate = False)
    for i in range(len(winners)):
        winners[i] = sr/taus[int(winners[i])]
    return winners
def step4(signal):
    """
    cumulative mean normalized differences with absolute threshold
    """
    winners = differences(signal, normalized=True, threshold=True, par_interpolation = False, local_estimate = False)
    for i in range(len(winners)):
        winners[i] = sr/taus[int(winners[i])]
    return winners
def step5(signal):
    """
    cumulative mean normalized differences with absolute threshold and parabolic interpolation
    """
    winners = differences(signal, normalized=True, threshold=True, par_interpolation = True, local_estimate = False)
    winners_frequencies = np.zeros_like(winners)
    for i,winner in enumerate(winners):
        winners_frequencies[i] = sr/(taus[int(winner)] * (1-winner%1) + taus[int(winner)+1] * (winner%1))
    return winners_frequencies
def step6(signal):
    """
    cumulative mean normalized differences with absolute threshold, parabolic interpolation and best local esitmate
    """
    winners = differences(signal, normalized=True, threshold=True, par_interpolation = True, local_estimate = True)
    winners_frequencies = np.zeros_like(winners)
    for i,winner in enumerate(winners):
        winners_frequencies[i] = sr/(taus[int(winner)] * (1-winner%1) + taus[int(winner)+1] * (winner%1))
    return winners_frequencies
def normalize_signal(signal):
    mean = np.mean(signal)
    std_dev = np.std(signal)
    signal = np.subtract(signal,mean)
    signal = np.multiply(signal,1/std_dev)
    return signal

y, sr = librosa.load('./sounds/audio_tone_440.wav', sr=None)
y = normalize_signal(y)

f_min = 80 #Hz
f_max = 1300 #Hz
taus = np.arange(start = sr//f_max, stop = sr//f_min)

coinv_window = int(taus[-1]+1)
n_windows = len(y)//coinv_window-1

pitches1 = step1(y)
pitches2 = step2(y)
pitches3 = step3(y)
pitches4 = step4(y)
pitches5 = step5(y)
pitches6 = step6(y)
librosa_pitches = librosa.yin(y,100,1300,win_length=coinv_window)
fig, axs = plt.subplots(3, 2)
custom_xlim = 0,len(y)/sr
custom_ylim = 0,max(pitches5)*1.25
axs[0, 0].plot(np.linspace(0, len(y)/sr, num = n_windows), pitches1)
axs[0, 0].hlines(440, -0.1*len(y)/sr,len(y)/sr*1.1, colors='r', linestyles='--')
axs[0, 0].set_title('Step 1')
axs[0, 1].plot(np.linspace(0, len(y)/sr, num = n_windows), pitches2)
axs[0, 1].hlines(440, -0.1*len(y)/sr,len(y)/sr*1.1, colors='r', linestyles='--')
axs[0, 1].set_title('Step 2')
axs[1, 0].plot(np.linspace(0, len(y)/sr, num = n_windows), pitches3)
axs[1, 0].hlines(440, -0.1*len(y)/sr,len(y)/sr*1.1, colors='r', linestyles='--')
axs[1, 0].set_title('Step 3')
axs[1, 1].plot(np.linspace(0, len(y)/sr, num = n_windows), pitches4)
axs[1, 1].hlines(440, -0.1*len(y)/sr,len(y)/sr*1.1, colors='r', linestyles='--')
axs[1, 1].set_title('Step 4')
axs[2, 0].plot(np.linspace(0, len(y)/sr, num = n_windows), pitches5)
axs[2, 0].hlines(440, -0.1*len(y)/sr,len(y)/sr*1.1, colors='r', linestyles='--')
axs[2, 0].set_title('Step 5')
axs[2, 1].plot(np.linspace(0, len(y)/sr, num = n_windows), pitches6)
axs[2, 1].hlines(440, -0.1*len(y)/sr,len(y)/sr*1.1, colors='r', linestyles='--')
axs[2, 1].set_title('Step 6')
for ax in axs.flat:
    ax.set(xlabel='t [sec]', ylabel='f [Hz]')

plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
plt.show()
