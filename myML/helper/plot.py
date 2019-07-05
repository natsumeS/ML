import os
import matplotlib.pyplot as plt
import time
import csv


class PlotHelper:
    def __init__(self, data_dir: str, base_filename: str, *, plot_show=False):
        self.fig, self.ax = plt.subplots(1, 1)
        self.x_list = []
        self.y_list = []
        self.line, = self.ax.plot(self.x_list, self.y_list)
        self.plot_show = plot_show
        self.dir = data_dir
        self.plot_filename = "{}/{}.png".format(self.dir, base_filename)
        self.csv_filename = "{}/{}.csv".format(self.dir, base_filename)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.time = None

    def add_data(self, x, y):
        self.x_list.append(x)
        self.y_list.append(y)
        self.line.set_data(self.x_list, self.y_list)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.savefig(self.plot_filename)
        if self.plot_show:
            plt.pause(0.1)

    def plot(self):
        plt.plot(self.x_list,self.y_list)

    def csv_recode_start(self, data):
        with open(self.csv_filename, "w") as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerow([0.0, data])
        self.time = time.time()

    def csv_add_data(self, data):
        with open(self.csv_filename, "a") as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerow([time.time() - self.time, data])
