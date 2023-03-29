import matplotlib.pyplot as plt
from easypyplot.barchart import draw
from easypyplot.pdf import plot_open
import os

data = [
    [1, 1, 1, 1,1,1,1,1],
    [1.1, 1.2, 0.9, 0.8,1,1,1,1],
    [0.8, 0.9, 2.0, 1.1,1,1,1,1],
    [1.0, 1.1, 1.2, 1.2,1,1,1,1],
    [0.8, 0.9, 2.0, 1.1,1,1,1,1],
    [1.0, 1.1, 1.2, 1.2,1,1,1,1],
]

nm1=["mnk","mkn","kmn","nmk","nkm","knm"]
nm2=[" I=1"," I=2"," I=3"," I=4"," I=5"," I=6"," I=7"," I=8"]
for i in range(0,6):
	for j in range(0,8):
		os.system("make "+nm1[i]+nm2[j])
		f=open("a.out","r")
		t=f.read()
		f.close()
		data[i][j]=float(t)
		print(data[i][j])
		print(str(i)+","+str(j))
for j in range(0,8):
	for i in range(1,6):
		data[i][j]=data[0][j]/data[i][j]
	data[0][j]=1

print(data)

with plot_open("lab") as fig:
    ax = fig.gca()
    draw(ax, data, breakdown=False,
            group_names=["mnk","mkn","kmn","nmk","nkm","knm"],
            entry_names=["1.txt", "2.txt", "3.txt", "4.txt","5.txt","6.txt","7.txt","8.txt"],
            )
    ax.set_xlabel('Different implementations of compute kernel')
    ax.set_ylabel('Speedup')
    ax.set_title('Task 1')
