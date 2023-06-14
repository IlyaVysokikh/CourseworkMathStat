from abc import ABC, abstractmethod
import numpy as np
import math
from typing import List, Tuple, Union


class Estimate(ABC):

    def __init__(self, sample: List[float]) -> None:
        self.sample = sample

    @abstractmethod
    def get(self) -> Union[float, Tuple[float, float]]:
        pass


class Core(ABC):
    @abstractmethod
    def _k(self, x: float) -> float:
        pass

    @abstractmethod
    def _K(self, x: float) -> float:
        pass

    @abstractmethod
    def h(self, x: float) -> float:
        pass


class FileManager:

    def __init__(self, path: str):
        self.path = path

    def read_sample(self) -> List[float]:
        try:
            with open(self.path) as file:
                sample = [float(x) for x in file.read().split()]
            return sample

        except FileNotFoundError:
            print("Файл не найден.")
            return []

        except ValueError:
            print("Неверный формат данных в файле.")
            return []

        except IOError:
            print("Произошла ошибка при чтении файла.")
            return []

    def write_sample(self, sample: List[float]) -> None:
        try:
            with open(self.path, 'w') as file:
                for value in sample:
                    file.write(str(value) + ' ')
            print("Выборка успешно записана в файл.")

        except IOError:
            print("Произошла ошибка при записи файла.")


class NormalCore(Core):
    def _k(self, x: float) -> float:
        return math.exp(-0.5 * x * x) / (math.sqrt(2 * math.pi))

    def _K(self, x: float) -> float:
        if x <= 0:
            return 0.852 * math.exp(-math.pow((-x + 1.5774) / 2.0637, 2.34))
        return 1 - 0.852 * math.exp(-math.pow((x + 1.5774) / 2.0637, 2.34))

    def h(self, x: float) -> float:
        return -x * (math.exp(-0.5 * x * x) / (math.sqrt(2 * math.pi)))


class Median(Estimate):

    def get(self) -> float:
        sample = self.sample
        sample.sort()
        if len(sample) % 2 == 0:
            return (sample[len(sample) // 2] + sample[len(sample) // 2 - 1]) / 2
        else:
            return sample[len(sample) // 2]


class Mean(Estimate):

    def get(self) -> float:
        return sum(self.sample) / len(self.sample)


class SampleVariance(Estimate):

    def get(self) -> float:
        mean = Mean(self.sample).get()
        return sum((x - mean) ** 2 for x in self.sample) / (len(self.sample) - 1)


class ConfidenceInterval(Estimate):

    def __init__(self, sample: List[float], core: Core, q: float):
        super().__init__(sample)
        self.core = core
        self.h = self._get_h()
        self.q = q

    def _get_t_critical(self) -> float:
        if self.q == 0.9:
            return 1.6602
        elif self.q == 0.95:
            return 1.9840
        else:
            return 2.6259

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

    def pdf(self, x: float) -> float:
        return np.mean([self.core._k((x - y) / self.h) for y in self.sample]) / self.h

    def get(self) -> Tuple[float, float]:
        median = Median(self.sample).get()

        t = self._get_t_critical()

        sigma = ((1 / (4 * (self.pdf(median) ** 2))) ** 0.5)
        tetta = t * sigma / math.sqrt(len(self.sample))

        lower_border = median - tetta
        upper_border = median + tetta
        print(tetta)
        return lower_border, upper_border


text = input().split()


sample = FileManager(text[0]).read_sample()

CI = ConfidenceInterval(sample, NormalCore(), float(text[1]))
print(CI.get())

