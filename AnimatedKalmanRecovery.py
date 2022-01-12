
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from scipy.linalg import inv

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

class KalmanFilter:
    
    def __init__(self, init_x = np.array([0., 0.])):
        
        self.dt = 1.
        self.x = init_x                   # Best to initialize, otherwise state starts at 0
        self.P = np.diag([4, 0.00000025]) # Covariance matrix; assume 𝜎 2 for measurement and 𝜎 500ppm for clock.
        self.F = np.array([[1, 1920.],    # State transition function; dt is chosen as the sample rate.
                           [0, 1    ]])

        self.Q = np.array([[0.588, 1.175],# Process noise; not currently used
                           [1.175, 2.35 ]])
        self.H = np.array([[1., 0.]])     # Measurement function
        self.R = np.array([[0.5]])        # Measurement noise; assume 𝜎 0.5 which is OK for semi-noisy input.

    def update(self, z):

        # predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T # + self.Q

        #update
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)

        # Normalize incoming sample point; this is needed
        # because the input wraps from 0->9 or 9->0.
        if (z - self.x[0] < -5):
            z += 10
        elif (z - self.x[0] > 5):
            z-= 10

        y = z - self.H @ self.x
        self.x += K @ y

        # Normalize the filtered sample point
        self.x[0] %= 10
        
        self.P = self.P - K @ self.H @ self.P
        return self.x

class AnimatedKalmanRecovery(object):

    def __init__(self, data):

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
        self.clock_line, = self.ax.plot(self.t, np.zeros(10), '-', label='Convolved')
        self.sample_line, = self.ax.plot(self.t, np.zeros(10), 'o', label='Convolved')
        
        plt.locator_params(axis='y', nbins=9)
        plt.locator_params(axis='x', nbins=10)
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        legend = plt.legend(bbox_to_anchor=(1.00, 0.5))
        # plt.legend()
        plt.close(self.fig)
        self.generator = clock_recovery(data)
        self.filter = None

    def init(self):

        return (self.symbol_line,self.conv_line, self.clock_line, self.sample_line, )

    def animate(self, i):

        n1, c1 = next(self.generator)

        i = np.argmax(c1)
        if self.filter is None:
            init_x = [float(i), 0.]
            self.filter = KalmanFilter(init_x)

        s,clk = self.filter.update(i)

        self.symbol_line.set_data(self.t, n1)
        self.conv_line.set_data(self.t, c1)
        self.clock_line.set_data(self.t, np.ones(10) * clk * 2000)
        sample = np.zeros(10)
        sample[:] = np.NaN
        sample[int(round(s)%10)] = 1
        self.sample_line.set_data(self.t, sample)

        return (self.symbol_line, self.conv_line, self.clock_line, self.sample_line, )

