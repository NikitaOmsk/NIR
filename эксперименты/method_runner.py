import numpy as np
import matplotlib.pyplot as plt
from time import time


class FirstOrderAlgorithmRunner:
    """
    Testing Method should return dict {
    'points': list of points,
    'grad_seq': list of gradient norm values,
    'func_seq': list of function values,
    'grad_count': total count of gradient calls,
    'func_count': total count of function calls}
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
        self.colors = ['b', 'r', 'g', 'y', 'c', 'm']

    def run_method(self, name, method, x0, func, grad, **parameters):
        time_start = time()
        result = method(x0, func, grad, self.verbose, **parameters)
        result['total_time'] = time() - time_start
        self.results[name] = result

    def norm(self, x):
        return np.sqrt((x ** 2).sum())

    def plot_points(self, capture, n_start=0, n_finish=-1, range_=None, projection=(0, 1)):
        xc, yc = projection
        legend = []
        for color, key in zip(self.colors, self.results):
            if range_:
                points = [point for point in self.results[key]['points'][n_start:n_finish] 
                          if range_[0]<=point[xc]<=range_[1] and range_[0]<=point[yc]<=range_[1]]
            else:
                points = self.results[key]['points'][n_start:]                    
            xs = [point[xc] for point in points]
            ys = [point[yc] for point in points]
            plt.scatter(x=xs, y=ys, lw=1, marker='.', c=color, edgecolors=color)
            legend.append(key)
        plt.legend(legend)
        plt.xlabel('$x_{'+str(xc)+'}$')
        plt.ylabel('$x_{'+str(yc)+'}$')
        plt.title(capture)

    def plot_distance(self, capture, n_start=0, n_finish=-1, x_target=None):
        from_last = (x_target is None)
        color_idx = 0
        legend = []
        for color, key in zip(self.colors, self.results):
            if from_last:
                x_target = self.results[key]['points'][-1]
            dist = [self.norm(point - x_target) for point in self.results[key]['points'][n_start:n_finish]]
            if log:
                dist = [np.log10(abs(g)) for g in dist]
            plt.scatter(x=range(n_start, n_start + len(dist)), y=dist, lw=1, marker='.', c=color, edgecolors=color)
            legend.append(key)
        plt.legend(legend)
        plt.xlabel('$n$')
        if log:
            plt.ylabel(r'$\lg||\Delta x_n||$')
        else:
            plt.ylabel(r'$||\Delta x_n||$')
        plt.title(capture)

    def plot_grads(self, capture, n_start=0, n_finish=-1, log=False):
        legend = []
        for color, key in zip(self.colors, self.results):
            grad = self.results[key]['grad_seq'][n_start:n_finish]
            if log:
                grad = [np.log10(abs(g)) for g in grad]
            plt.scatter(x=range(n_start, n_start + len(grad)), y=grad, lw=1, marker='.', c=color, edgecolors=color)
            legend.append(key)
        plt.legend(legend)
        plt.xlabel('$n$')
        if log:
            plt.ylabel(r'$\lg||\nabla f(x_n)||$')
        else:
            plt.ylabel(r'$||\nabla f(x_n)||$')
        plt.title(capture)

    def plot_func(self, capture, n_start=0, n_finish=-1, log=False):
        color_idx = 0
        legend = []
        for color, key in zip(self.colors, self.results):
            fval = self.results[key]['func_seq'][n_start:n_finish]
            if log:
                fval = [np.log10(abs(g)) for g in fval]
            plt.scatter(x=range(n_start, n_start + len(fval)), y=fval, lw=1, marker='.', c=color, edgecolors=color)
            legend.append(key)
        plt.legend(legend)
        plt.xlabel('$n$')
        if log:
            plt.ylabel(r'$\lg|f(x_n)|$')
        else:
            plt.ylabel(r'$|f(x_n)|$')
        plt.title(capture)

    def print_data(self):
        for key in self.results:
            print("Experiment '{}': "
                  "total time {} seconds;\n"
                  "gradient called {} times, "
                  "function called {} times;\n"
                  "minimal gradient norm is {}, "
                  "minimal function value is {}".format(key,
                                                        self.results[key]['total_time'],
                                                        self.results[key]['grad_count'],
                                                        self.results[key]['func_count'],
                                                        min(self.results[key]['grad_seq']),
                                                        min(self.results[key]['func_seq'])))
