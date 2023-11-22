import matplotlib.pyplot as plt

sciezka_do_pliku1 = 'beforeTraining.txt'
sciezka_do_pliku2 = 'afterTraining.txt'

with open(sciezka_do_pliku1, 'r') as plik1, open(sciezka_do_pliku2, 'r') as plik2:
    lines1 = plik1.readlines()
    lines2 = plik2.readlines()

liczba_detekcji1 = [int(line) for line in lines1]
liczba_detekcji2 = [int(line) for line in lines2]

czas = [i for i in range(1, len(liczba_detekcji1) + 1)]

plt.plot(czas, liczba_detekcji1, label='przed treningiem')
plt.plot(czas, liczba_detekcji2, label='po treningu')
plt.xlabel('Czas (s)')
plt.ylabel('Liczba detekcji')
plt.title('Zmiana liczby detekcji w czasie trwania nagrania dla modelu YOLOv5s')
plt.legend()
plt.grid(True)
plt.xticks(range(5, len(liczba_detekcji1) + 1, 5))
plt.show()
