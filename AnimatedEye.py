import numpy as np
import matplotlib.pyplot as plt


class AnimatedEye:
    
    def __init__(self, data):

        self.SYMBOLS = 192
        self.SAMPLES = 1920
        self.samples_per_symbol = int(self.SAMPLES / self.SYMBOLS)
        self.data = data[:(len(data) // 1920) * 1920].reshape(-1,1920)
        self.t = np.linspace(0, 10, 10, endpoint=False)
        plt.figure()
        plt.rcParams['figure.figsize'] = [9, 4]

        self.fig, self.ax = plt.subplots(1, sharex=True)
        plt.xlabel('Symbol Periods')
        plt.title("Eye Diagram")
        plt.rcParams["animation.html"] = "jshtml"
        plt.rcParams["animation.embed_limit"] = 20.0
        self.ax.grid(which='major', alpha=0.5)
        self.ax.set_ylim(-1.2, 1.2)

        plt.locator_params(axis='y', nbins=9)
        plt.locator_params(axis='x', nbins=10)
        
        eye = self.data[0].reshape((-1, self.samples_per_symbol))
        self.symbol_line = self.ax.plot(self.t, eye[:self.SYMBOLS,:].T, color = 'C0', alpha = 0.1)
        plt.close(self.fig)
        
    def init(self):

        return (*self.symbol_line,)

    def animate(self, i):
                                   
        eye = self.data[i].reshape((-1, self.samples_per_symbol))
        for l, e in zip(self.symbol_line,  eye[:self.SYMBOLS,:]):
            l.set_data(self.t, e)
        return (*self.symbol_line,)

