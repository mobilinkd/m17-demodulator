
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage

def clock_recovery(data):
    
    dt = np.concatenate((
        [0],
        (data[1:] - data[:-1]) * np.array([-1 if d1 + d2 < 0 else 1 for d1, d2 in zip(data[1:], data[:-1])])
    ))

    impulse = np.sin(np.linspace(np.pi/-2, np.pi/2, 5))
    frames = dt.reshape(-1,1920)
    s_frames = [np.reshape(f, (192, 10)) for f in frames]
    for frame in s_frames:
        n = np.apply_along_axis(np.sum, 0, frame)
        n1 = n / n.max() # Normalize
        c = ndimage.convolve(n1, impulse, mode="wrap")
        c1 = c / c.max() # Normalize
        yield n1, c1

class AnimatedClockRecovery(object):

    def __init__(self, data):

        dt = (data[1:] - data[:-1]) * np.array([-.00001 if d1 + d2 < 0 else .00001 for d1,d2 in zip(data[1:], data[:-1])])
        self.dt = dt[:(len(dt)//1920)*1920].reshape(-1,1920)
        self.data = data[1:(len(dt)//1920)*1920 + 1].reshape(-1,1920)
        
        self.t = np.linspace(0, 1, 10, endpoint=False)
        plt.figure()
        plt.rcParams['figure.figsize'] = [9, 4]

        self.signal = np.sin(np.linspace(np.pi/-2, np.pi/2, 5))

        self.fig, self.ax = plt.subplots(1, sharex=True)
        plt.xlabel('Symbol Periods')
        plt.title("Clock Recovery")
        plt.rcParams["animation.html"] = "jshtml"
        plt.rcParams["animation.embed_limit"] = 20.0
        self.ax.grid(which='major', alpha=0.5)
        self.ax.set_ylim(-1.2, 1.2)

        self.symbol_line, = self.ax.plot(self.t, np.zeros(10), '-', label='Symbol')
        self.conv_line, = self.ax.plot(self.t, np.zeros(10), '-', label='Convolved')
        
        plt.locator_params(axis='y', nbins=9)
        plt.locator_params(axis='x', nbins=10)
        legend = plt.legend(bbox_to_anchor=(0.65, 0.7, 0.5, 0.5))
        # plt.legend()
        plt.close(self.fig)
        self.generator = clock_recovery(data)

    def init(self):

        return (self.symbol_line,self.conv_line,)

    def animate(self, i):

        
        p = np.reshape(self.data[i], (192, 10))
        d = np.reshape(self.dt[i], (192, 10))
        n = np.apply_along_axis(np.sum, 0, d)
        n1 = n / n.max()
        c = ndimage.convolve(n1, self.signal, mode="wrap")
        c1 = c / c.max()

        n1, c1 = next(self.generator)

        self.symbol_line.set_data(self.t, n1)
        self.conv_line.set_data(self.t, c1)
        return (self.symbol_line, self.conv_line, )

