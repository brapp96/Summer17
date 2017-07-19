from src import VEClib

for i in [100,200,500,1000,2000,5000,10000]:
    f = VEClib.plot_res(i, ['deep','nbt'], i)
    f.show()
raw_input()
