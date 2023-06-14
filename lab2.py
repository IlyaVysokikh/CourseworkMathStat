import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import numpy as np
from tkinter import *
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

class RandomVariable(ABC):
  @abstractmethod
  def pdf(self, x):
    pass

  @abstractmethod
  def cdf(self, x):
    pass

  @abstractmethod
  def quantile(self, alpha):
    pass


class Core(ABC):
    @abstractmethod
    def _k(self, x):
        pass

    @abstractmethod
    def _K(self, x):
        pass

    @abstractmethod
    def h(self, x):
        pass


class RandomNumberGenerator(ABC):
  def __init__(self, random_variable: RandomVariable):
    self.random_variable = random_variable

  @abstractmethod
  def get(self, N):
    pass


class Estimation(ABC):
    def __init__(self, sample):
        self.sample = sample


class Mean(Estimation):

    def get(self) -> float:
        return sum(self.sample) / len(self.sample)


class SampleVariance(Estimation):

    def get(self) -> float:
        mean = Mean(self.sample).get()
        return sum((x - mean) ** 2 for x in self.sample) / (len(self.sample) - 1)


class SimpleRandomNumberGenerator(RandomNumberGenerator):
  def __init__(self, random_variable):
    super().__init__(random_variable)

  def get(self, N):
    us = np.random.uniform(0, 1, N)
    return np.vectorize(self.random_variable.quantile)(us)


class NormalRandomVariable(RandomVariable):
  def __init__(self, location=0, scale=1) -> None:
    super().__init__()
    self.location = location
    self.scale = scale

  def pdf(self, x):
    z = (x - self.location) / self.scale
    return math.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * self.scale)

  def cdf(self, x):
    z = (x - self.location) / self.scale
    if z <= 0:
      return 0.852 * math.exp(-math.pow((-z + 1.5774) / 2.0637, 2.34))
    return 1 - 0.852 * math.exp(-math.pow((z + 1.5774) / 2.0637, 2.34))

  def quantile(self, alpha):
    return self.location + 4.91 * self.scale * (math.pow(alpha, 0.14) - math.pow(1 - alpha, 0.14))

class UniformRandomVariable(RandomVariable):
  def __init__(self, a=0, b=1) -> None:
    super().__init__()
    self.a = a
    self.b = b

  def pdf(self, x):
    if x >= self.a and x <= self.b:
      return 1 / (self.b - self.a)
    else:
      return 0

  def cdf(self, x):
    if x <= self.a:
      return 0
    elif x >=self.b:
      return 1
    else:
      return (x - self.a) / (self.b - self.a)

    def quantile(self, alpha):
      return self.a + alpha * (self.b - self.a)

class ExponentialRandomVariable(RandomVariable):
    def __init__(self, rate = 1):
      self.rate = rate

    def pdf(self, x):
      if x < 0:
        return 0
      else:
        return self.rate * math.exp(-self.rate * x)

    def cdf(self, x):
      if x < 0:
        return 0
      else:
        return 1 - math.exp(-self.rate * x)

    def quantile(self, alpha):
      return -math.log(1 - alpha) / self.rate


class LaplaceRandomVariable(RandomVariable):
    def __init__(self, loc=0, scale=1):
      self.loc = loc
      self.scale = scale

    def pdf(self, x):
      return 0.5 * self.scale * math.exp(-self.scale * abs(x - self.loc))

    def cdf(self, x):
      if x < self.loc:
        return 0.5 * math.exp((x - self.loc) / self.scale)
      else:
        return 1 - 0.5 * math.exp(-(x - self.loc) / self.scale)

    def quantile(self, alpha):
      if alpha == 0.5:
        return self.loc
      elif alpha < 0.5:
        return self.loc - self.scale * math.log(1 - 2 * alpha)
      else:
        return self.loc + self.scale * math.log(2 * alpha - 1)

class CauchyRandomVariable(RandomVariable):
    def __init__(self, loc=0, scale=1):
      self.loc = loc
      self.scale = scale

    def pdf(self, x):
      return 1 / (math.pi * self.scale * (1 + ((x - self.loc) / self.scale) ** 2))

    def cdf(self, x):
      return 0.5 + math.atan((x - self.loc) / self.scale) / math.pi

    def quantile(self, alpha):
      return self.loc + self.scale * math.tan(math.pi * (alpha - 0.5))


class NormalCore(Core):
    def _k(x):
        z = x
        return math.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi))

    def _K(x):
        z = x
        if z <= 0:
            return 0.852 * math.exp(-math.pow((-z + 1.5774) / 2.0637, 2.34))
        return 1 - 0.852 * math.exp(-math.pow((z + 1.5774) / 2.0637, 2.34))

    def h(x):
        return -x * (math.exp(-0.5 * x * x) / (math.sqrt(2 * math.pi)))


class Histogram(Estimation):
    class Interval:
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def is_in(self, x):
            return x >= self.a and x <= self.b

        def __repr__(self):
            return f'({self.a}, {self.b})'

    def __init__(self, sample, m):
        super().__init__(sample)
        self.m = m

        self.init_intervals()

    def init_intervals(self):
        left_boundary_of_intervals = np.linspace(np.min(self.sample), np.max(self.sample), self.m + 1)[:-1]
        right_boundary_of_intervals = np.concatenate((left_boundary_of_intervals[1:], [np.max(self.sample)]))

        self.intervals = [Histogram.Interval(a, b) for a, b in
                          zip(left_boundary_of_intervals, right_boundary_of_intervals)]

        self.sub_interval_width = right_boundary_of_intervals[0] - left_boundary_of_intervals[0]

    def get_interval(self, x):
        for i in self.intervals:
            if i.is_in(x):
                return i
        return None

    def get_sample_by_interval(self, interval):
        return np.array(list(filter(lambda x: interval.is_in(x), self.sample)))

    def value(self, x):
        return len(self.get_sample_by_interval(self.get_interval(x))) / (self.sub_interval_width * len(self.sample))


class EDF(Estimation):
    def heaviside_function(x):
        if x > 0:
            return 1
        else:
            return 0

    def value(self, x):
        return np.mean(np.vectorize(EDF.heaviside_function)(x - self.sample))


class SmoothedRandomVariable(RandomVariable, Estimation):
    def __init__(self, sample, core: Core):
        super().__init__(sample)
        self.core = core
        self.h = self._get_h()


    def _get_h(self) -> float:
        self.h = math.sqrt(SampleVariance(self.sample).get())

        accuracy = 1

        while accuracy >= 0.001:
            s = 0
            for i, sample_i in enumerate(self.sample):
                num = 0
                den = 0
                for j, sample_j in enumerate(self.sample):
                    if i == j:
                        continue
                    difference = (sample_j - sample_i) / self.h
                    k2 = self.core.h(difference)
                    k1 = self.core._k(difference)
                    num += k2 * (sample_j - sample_i)
                    den += k1
                s += num / den
            new_std_deviation = -s / len(self.sample)
            accuracy = abs(new_std_deviation - self.h)
            self.h = new_std_deviation

        return self.h

    def pdf(self, x):
        return np.mean([self.core._k((x - y) / self.h) for y in self.sample]) / self.h

    def cdf(self, x):
        # h = self.get_h()
        return np.mean([self.core._K((x - y) / self.h) for y in self.sample])

    def quantile(self, alpha):
        raise NotImplementedError


def plot(xs, ys, colors):
    for x, y, c in zip(xs, ys, colors):
        plt.plot(x, y, c)
    plt.show()


root = Tk()
root.resizable(width=False, height=False)
WIDTH = 1100
HEIGHT = 750
root.geometry("1100x750")


def save_config():
    save_config_button.config(state="normal")
    unblock_text()
    start_button.config(state="normal")
    clear_text()
    if select_of_distribtions.get() == "Нормальное":
        labelX1.config(text="μ")
        enterX1.insert(0, "0")
        labelX2.config(text="σ")
        enterX2.insert(0, "1")
    elif select_of_distribtions.get() == "Равномерное":
        labelX1.config(text="a")
        enterX1.insert(0, "0")
        labelX2.config(text="b")
        enterX2.insert(0, "1")
    elif select_of_distribtions.get() == "Экспоненциальное":
        labelX1.config(text="ʎ")
        enterX1.insert(0, "1")
        labelX2.config(text="-")
        enterX2.config(state="disabled")
    elif select_of_distribtions.get() == "Коши":
        labelX1.config(text="x0")
        enterX1.insert(0, "0")
        labelX2.config(text="γ")
        enterX2.insert(0, "1")
    else:
        labelX1.config(text="β")
        enterX1.insert(0, "0")
        labelX2.config(text="α")
        enterX2.insert(0, "1")


def unblock_text():
    enterN.config(state="normal")
    enterX1.config(state="normal")
    enterX2.config(state="normal")
    # enterBandWidth.config(state="normal")
    enterM.config(state="normal")


def block_text():
    enterN.config(state="disabled")
    enterX1.config(state="disabled")
    enterX2.config(state="disabled")
    # enterBandWidth.config(state="disabled")
    enterM.config(state="disabled")


def clear_text():
    enterX1.delete(0, END)
    enterX2.delete(0, END)


def start():
    labelFrame = Label(root,
                       text="Функция распределения (красный),\nэмпирическая функция распределения (синий), \nсглаженная эмпирическая (зеленый)")
    labelFrame.place(x=50, y=200)
    labelFrame2 = Label(root, text="истинная плотность (красный)\nгистограмма (синий)\nоценка плотности Розенблатта-Парзена")
    labelFrame2.place(x=600, y=200)
    N = int(enterN.get())
    # rv = db.NormalRandomVariable
    if select_of_distribtions.get() == "Нормальное":
        rv = NormalRandomVariable(int(enterX1.get()), int(enterX2.get()))
    elif select_of_distribtions.get() == "Равномерное":
        rv = UniformRandomVariable(int(enterX1.get()), int(enterX2.get()))
    elif select_of_distribtions.get() == "Экспоненциальное":
        rv = ExponentialRandomVariable(int(enterX1.get()))
    elif select_of_distribtions.get() == "Коши":
        rv = CauchyRandomVariable(int(enterX1.get()), int(enterX2.get()))
    else:
        rv = LaplaceRandomVariable(int(enterX1.get()), int(enterX2.get()))
    generator = SimpleRandomNumberGenerator(rv)
    sample = generator.get(N)
    M = 100
    X = np.linspace(np.min(sample), np.max(sample), M)
    Y_truth = np.vectorize(rv.cdf)(X)
    edf = EDF(sample)
    Y_edf = np.vectorize(edf.value)(X)
    # bandwidth = float(enterBandWidth.get())



    srv = SmoothedRandomVariable(sample, NormalCore)
    Y_kernel = np.vectorize(srv.cdf)(X)
    global canvas1
    global canvas2
    ax.clear()
    for x, y, c in zip([X] * 3, [Y_truth, Y_edf, Y_kernel], ['r', 'b', 'g']):
        ax.plot(x, y, c)
    if canvas1 is not None:
        canvas1.get_tk_widget().pack_forget()
        canvas1.get_tk_widget().destroy()
    canvas1 = FigureCanvasTkAgg(fig, master=frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack()

    # plot([X]*3, [Y_truth, Y_edf, Y_kernel], ['r', 'b', 'g'])

    P_1 = np.vectorize(rv.pdf)(X)
    m = int(enterM.get())

    hist = Histogram(sample, m)
    P_2 = np.vectorize(hist.value)(X)
    # plot([X]*2, [P_1, P_2], ['r', 'b'])

    P_3 = np.vectorize(srv.pdf)(X)
    graphics.clear()
    for x, y, c in zip([X] * 3, [P_1, P_2, P_3], ['r', 'b', 'g']):
        graphics.plot(x, y, c)

    if canvas2 is not None:
        canvas2.get_tk_widget().pack_forget()
        canvas2.get_tk_widget().destroy()
    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas2.draw()
    canvas2.get_tk_widget().pack()
    # plot([X]*3, [P_1, P_2, P_3], ['r', 'b', 'g'])


canvas1 = None
canvas2 = None

distrubtions = ["Нормальное", "Равномерное", "Экспоненциальное", "Лапласса", "Коши"]
select_of_distribtions = ttk.Combobox(values=distrubtions, state="readonly")
select_of_distribtions.pack(anchor="se", padx=75, pady=5)
select_of_distribtions.current(0)
# select_of_distribtions.state = "disabled"
# select_distr_label = Label(root, text="Выберите распределение", fg="black")
# select_distr_label.place(x=600, y=10, height=12, width=180)
distrubtions_label = Label(root, text="Вы выбрали: " + select_of_distribtions.get(), fg="black")

# cores = ["Епанечникова", "Треугольника", "Прямоугольника", "Нормальное"]
# select_of_cores = ttk.Combobox(values=cores, state="readonly")
# select_of_cores.pack(anchor="nw", padx=5, pady=5)
# select_of_cores.current(0)
select_core_label = Label(root, text="Ядро: нормальное", fg="black", bg="white")
select_core_label.place(x=880, y=50, height=18, width=110)
core_label = Label(root, text="Ядро: нормальное", fg="black")

save_config_button = Button(root, text="Выбрать", command=save_config)
save_config_button.place(x=880, y=90)


enterN = Entry()
enterN.place(x=40, y=700)
enterN.insert(0, "10")
enterN.config(state="normal")
labelN = Label(root, text="N")
labelN.place(x=40, y=670)

enterX1 = Entry()
enterX1.place(x=200, y=700)
enterX1.insert(0, "0")
enterX1.config(state="normal")
labelX1 = Label(root, text="μ")
labelX1.place(x=200, y=670)

enterX2 = Entry()
enterX2.place(x=360, y=700)
enterX2.insert(0, "1")
enterX2.config(state="normal")
labelX2 = Label(root, text="σ")
labelX2.place(x=360, y=670)

# enterBandWidth = Entry()
# enterBandWidth.place(x=520, y=700)
# enterBandWidth.insert(0, "0.5")
# enterBandWidth.config(state="normal")
# labelBandWidth = Label(root, text="bw")
# labelBandWidth.place(x=520, y=670)

enterM = Entry()
enterM.place(x=520, y=700)
enterM.insert(0, "20")
enterM.config(state="normal")
labelM = Label(root, text="m")
labelM.place(x=520, y=670)

frame = Frame(root)
frame.place(x=50, y=250)
fig = plt.Figure(figsize=(4.5, 3.5), dpi=100)
ax = fig.add_subplot(111)

frame2 = Frame(root)
frame2.place(x=600, y=250)
fig2 = plt.Figure(figsize=(4.5, 3.5), dpi=100)
graphics = fig2.add_subplot(111)


start_button = Button(root, state="normal", text="Запустить", command=start, font=("Arial", 18))
start_button.place(x=840, y=690)

root.mainloop()
