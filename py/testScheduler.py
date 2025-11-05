import launchScheduler as lsch
from   scipy.io   import  loadmat
import numpy      as      np

# Загрузить матрицу запусков из .mat файла
f = loadmat( "launchData.mat" )
A = f.get('scheduleMatrix')

# Сохранить матрицу запусков в отельный файл
np.save("scheduleMatrix", A)

# Инициализация
sch = lsch.LaunchScheduler("scheduleMatrix.npy", "newScheduleMatrix.npy", 14, 12, 24)

# Найти новую матрицу запусков и сохранить результат в заданный файл
sch.solve()

# Распечатка результата
newsch = np.load("newScheduleMatrix.npy")
print(newsch)