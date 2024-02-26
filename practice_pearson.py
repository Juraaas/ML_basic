import pandas as pd
import matplotlib.pyplot as plt
import matplotlib_inline
import math

df = pd.DataFrame()
df['X'] = [1, 2, 3, 4, 5]
df['Y'] = [4, 6, 9, 11, 18]
print(df)
plt.scatter(df['X'], df['Y'], label='Wartosci')
plt.xlabel('Wartosci X')
plt.ylabel('Wartosci Y')
plt.legend()
plt.show()

n = len(df['X'])
pearson = pd.DataFrame(df[:])
pearson['y2'] = df['Y'] * df['Y']
pearson['xy'] = df['X'] * df['Y']
pearson['x2'] = df['X'] * df['X']
pearson.loc['sum'] = pearson.sum()
print()
print(pearson)


def srednia(zbior):
    return float(sum(zbior) / len(zbior))


def odchylenie(zbior):
    wynik = 0
    for i in range(len(zbior)):
        wynik += (zbior[i] - srednia(zbior)) ** 2
    return math.sqrt(wynik / len(zbior - 1))


def pearson_kor(zbior1):
    return (len(zbior1) * sum(pearson['xy']) - (sum(pearson['X']) * sum(pearson['Y']))
            / math.sqrt(len(zbior1) * sum(pearson['x2']) - (sum(pearson['X'] ** 2)))
            * (len(zbior1) * sum(pearson['y2']) - (sum(pearson['Y']) ** 2)))


Sx = odchylenie(df['X'])
Sy = odchylenie(df['Y'])
#print("Odchylenie standardowe x: ", Sx)
#print("Odchylenie standardowe y: ", Sy)


mean_x = srednia(df['X'])
mean_y = srednia(df['Y'])
#print("Srednia x:", mean_x)
#print("Srednia y:", mean_y)
kor = pearson_kor(df['X'])
print("Wspol kor zbioru: ", kor)