from random import choice
from Levenshtein import distance  # NOQA
import numpy as np
import time

# parametry instancji
N = int(500)
K = int(9)
E = float(0.1)
MINOVERLAP = int(3)  # maksymalna akcepotwalna waga krawedzi grafu

# procent szans na zignorowanie, ze dany wierzcholek jest mozliwym bledem pozytywnym
IGN_POS_ERR = int(5)
GENERATION_SIZE = int(100)
MAX_TIME_SECONDS = int(90)

# opoznienie w iteracjach po ktorym zaczyna sie zwiekszac szansa na uzycie macierzy feromonowej
USE_PHEROMONES_DELAY = int(10)
WSP_PAROWANIA = float(0.9)  # procent feromonu jaki zostaje po parowaniu

NUCLEOTIDES = ["A", "C", "G", "T"]


class SBH:
    def __init__(self, n: int, k: int, e: float, minOverlap: int) -> None:
        self.n = n
        self.k = k
        self.e = e
        self.perfectSpectrumLenght = self.n - self.k + 1
        self.numberOfErrors = int(self.perfectSpectrumLenght * self.e)
        self.minOverlap = minOverlap
        self.seq: str = ""
        self.spectrum: list = []
        self.begin: str = ""
        self.matrix = []
        self.posErrors = []
        self.listNast = []
        self.__generateSequence()
        self.__generateSpectrum()
        self.begin = self.spectrum[0]
        self.__addErrors()
        self.spectrum.sort()
        self.__generateMatrix()
        self.__findPosErrors()
        self.__generateList()

    def __generateSequence(self):
        for _ in range(self.n):
            self.seq += np.random.choice(NUCLEOTIDES)

    def __generateSpectrum(self):
        for i in range(self.perfectSpectrumLenght):
            newOligo = self.seq[i : i + self.k]
            if newOligo not in self.spectrum:
                self.spectrum.append(newOligo)

    def __addErrors(self):
        # bledy pozytywne
        if self.numberOfErrors == 0:
            return
        while len(self.spectrum) != self.perfectSpectrumLenght - self.numberOfErrors:
            toDelete = choice(self.spectrum)
            if toDelete == self.begin:
                continue
            else:
                self.spectrum.remove(toDelete)

        # bledy negatywne
        for _ in range(self.numberOfErrors):
            newOligo = "".join(np.random.choice(NUCLEOTIDES, self.k))
            self.spectrum.append(newOligo)

    def __generateMatrix(self):
        self.matrix = [[0 for _ in self.spectrum] for _ in self.spectrum]
        for i in range(len(self.spectrum)):
            for j in range(len(self.spectrum)):
                for overlap in range(self.minOverlap, 0, -1):
                    if (
                        i != j
                        and self.spectrum[i][overlap:] == self.spectrum[j][:-overlap]
                    ):
                        self.matrix[i][j] = overlap

    def __findPosErrors(self) -> None:
        if self.numberOfErrors == 0:
            return
        inDegree = np.count_nonzero(self.matrix, axis=0)
        outDegree = np.count_nonzero(self.matrix, axis=1)
        verticeDegree = np.add(inDegree, outDegree)
        verticeDegree[self.spectrum.index(self.begin)] = 999
        tmp = sorted(zip(self.spectrum, verticeDegree), key=lambda item: item[1])
        for i in range(self.numberOfErrors):
            self.posErrors.append(self.spectrum.index(tmp[i][0]))

    def __generateList(self) -> None:
        self.listNast = [[] for _ in range(len(self.spectrum))]

        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if self.matrix[i][j] > 0:
                    self.listNast[i].append((j, self.matrix[i][j]))


class Solution:
    def __init__(
        self, instance: SBH, path=[], seq="", oneWeightTaken=0, posErrorsTaken=0
    ) -> None:
        self.path = path
        self.seq = seq
        self._oneWeightTaken = oneWeightTaken
        self._posErrorsTaken = posErrorsTaken
        self._perfSpectrum = instance.n - instance.k + 1
        self._iloscBledow = self._perfSpectrum * instance.e
        self.quality = 0.0
        self.__calculateQuality()

    def __calculateQuality(self):
        self.quality = (2 * self._oneWeightTaken + len(self.path)) / (
            3 * self._perfSpectrum
        )

    def __repr__(self) -> str:
        return f"{self.quality}"


def roulette_wheel_selection(items: tuple, reverse=True) -> any:
    if reverse:
        chances = [1 / item[1] for item in items]
        population_fitness = sum(chances)
    else:
        chances = [item[1] for item in items]
        population_fitness = sum(chances)

    chromosome_probabilities = [chance / population_fitness for chance in chances]

    return items[np.random.choice(len(items), p=chromosome_probabilities)]


def algorytmLosowy(a: SBH) -> Solution:
    wynik = a.begin
    path = []
    oneWeightCounter = 0
    posErrorCounter = 0
    beginId = a.spectrum.index(a.begin)
    firstPass = True

    curVertice = (beginId, 1)

    while len(wynik) < N:
        possibleVertices = []

        for vertice in a.listNast[curVertice[0]]:
            if vertice[0] in path or not a.listNast[vertice[0]]:
                continue
            elif (
                vertice[0] in a.posErrors
                and roulette_wheel_selection(
                    [(0, 100 - IGN_POS_ERR), (1, IGN_POS_ERR)]
                )[0]
            ):
                possibleVertices.append(vertice)
                posErrorCounter += 1
            elif vertice[0] not in a.posErrors:
                possibleVertices.append(vertice)

        if not possibleVertices:
            # jesli nie ma mozliwych wierzcholkow do odwiedzenia
            try:
                backCounter = 0
                while (
                    len(a.listNast[path[-(backCounter + 1)]]) < 2
                    or roulette_wheel_selection([(0, 2), (1, 1)])[0]
                ):
                    backCounter += 1
            except IndexError:
                print("brak mozliwych wierzcholkow do odwiedzenia")
                return Solution(a, path, wynik, oneWeightCounter, posErrorCounter)

            for _ in range(backCounter):
                path.pop()

            possibleVertices.append(roulette_wheel_selection(a.listNast[path[-1]]))

        nextVertice = roulette_wheel_selection(possibleVertices)
        wynik += a.spectrum[nextVertice[0]][
            a.k - a.matrix[curVertice[0]][nextVertice[0]] :
        ]
        if firstPass:
            wynik = (
                wynik[: a.k] + wynik[a.k + a.matrix[curVertice[0]][nextVertice[0]] :]
            )
            firstPass = False
        curVertice = nextVertice

        path.append(curVertice[0])
        if curVertice[1] == 1:
            oneWeightCounter += 1
    return Solution(a, path, wynik, oneWeightCounter, posErrorCounter)


def antPass(a: SBH, pheromones: list, usePheromonesChance: float) -> Solution:
    wynik = a.begin
    path = []
    oneWeightCounter = 1
    posErrorCounter = 0
    beginId = a.spectrum.index(a.begin)
    usePheromonesCounter = 0
    firstPass = True

    curVertice = (beginId, float(1))

    while len(wynik) < N:
        possibleVertices = list()
        usePheromones = roulette_wheel_selection(
            [(0, 100 - usePheromonesChance), (1, usePheromonesChance)], reverse=False
        )[0]
        if not usePheromones:
            for vertice in a.listNast[curVertice[0]]:
                if vertice[0] in path or not a.listNast[vertice[0]]:
                    continue
                elif (
                    vertice[0] in a.posErrors
                    and roulette_wheel_selection(
                        [(0, 100 - IGN_POS_ERR), (1, IGN_POS_ERR)]
                    )[
                        0
                    ]  # NOQA
                ):
                    possibleVertices.append(vertice)
                    posErrorCounter += 1
                elif vertice[0] not in a.posErrors:
                    possibleVertices.append(vertice)

            for vertice in possibleVertices:
                if vertice[1] == 0:
                    possibleVertices.remove(vertice)

            if not possibleVertices:
                return Solution(a, path, wynik, oneWeightCounter, posErrorCounter)

            nextVertice = roulette_wheel_selection(possibleVertices)
        else:
            usePheromonesCounter += 1
            for i, item in enumerate(pheromones[curVertice[0]]):
                if item and item not in path:
                    possibleVertices.append((i, item))

            if not possibleVertices:
                return Solution(a, path, wynik, oneWeightCounter, posErrorCounter)
            nextVertice = roulette_wheel_selection(possibleVertices, reverse=False)

        overlap = a.matrix[curVertice[0]][nextVertice[0]]
        if overlap == 1:
            oneWeightCounter += 1
        if not firstPass:
            wynik += a.spectrum[nextVertice[0]][-overlap:]

        curVertice = nextVertice
        path.append(curVertice[0])
        firstPass = False
    return Solution(a, path, wynik, oneWeightCounter, posErrorCounter)


def antColonyOptimization(a: SBH):
    interationCount: int = int(0)
    elapsedTime: int = int(0)
    usePheromonesChance: float = 0.0
    pheromones: list[list[float]] = [[0.0 for _ in a.spectrum] for _ in a.spectrum]
    bestSolution = Solution(a)

    startTime = time.process_time()
    while elapsedTime < MAX_TIME_SECONDS:
        interationCount += 1
        generation: list[Solution] = []

        # aktualizacja szansy użycia feromonów
        if usePheromonesChance < 100 and interationCount > USE_PHEROMONES_DELAY:
            usePheromonesChance += 0.3
            usePheromonesChance *= 1.1
            if usePheromonesChance > 100:
                usePheromonesChance = 100

        # tworzenie generacji 100 mrówek
        for _ in range(GENERATION_SIZE):
            generation.append(antPass(a, pheromones, usePheromonesChance))

        # sortowanie wyników wg wartości fukcji Cmax
        generation.sort(key=lambda x: x.quality, reverse=True)

        # zapisywanie najlepszego rozwiazania
        if bestSolution.quality < generation[0].quality:
            bestSolution = generation[0]
            print(
                f"{interationCount} : {distance(bestSolution.seq, a.seq)} : {bestSolution.quality}"
            )

        # parowanie macierzy feromonowej
        for i in range(len(pheromones)):
            for j in range(len(pheromones[i])):
                pheromones[i][j] = pheromones[i][j] * WSP_PAROWANIA

        # dodwanie warotsci do macierzy feromonowej
        pheromoneValue = 1
        for i in range(int(len(generation) * 0.1)):
            for j in range(1, len(generation[i].path)):
                pheromones[generation[i].path[j - 1]][
                    generation[i].path[j]
                ] += pheromoneValue
            pheromoneValue -= 0.1
        elapsedTime = time.process_time() - startTime
    # print("iteracji: ", interationCount)
    return bestSolution


def main():
    test_instance = SBH(N, K, E, MINOVERLAP)
    wynikACO = antColonyOptimization(test_instance)

    print("\nNajlepszy wynik:")
    print("Quality:", wynikACO.quality)
    print("Distance:", distance(wynikACO.seq, test_instance.seq))


if __name__ == "__main__":
    main()
