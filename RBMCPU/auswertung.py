import numpy as np
import matplotlib.pyplot as plt

data1 = np.genfromtxt("weights_ising_layer_0.csv",delimiter=",")
maxes = []
max1 = np.max(np.abs(data1))
divider = max1
maxes += [max1/divider]
data1 = data1 / divider
plt.imshow(data1, cmap="hot")
plt.colorbar()
plt.show()
data2 = np.genfromtxt("weights_ising_layer_1.csv",delimiter=",")
data2 = data2/divider
maxes += [np.max(np.abs(data2))]
#plt.imshow(data2, cmap="hot")
#plt.colorbar()
#plt.show()
data3 = np.genfromtxt("weights_ising_layer_2.csv",delimiter=",")
data3 = data3 / divider
maxes += [np.max(np.abs(data3))]
data4 = np.genfromtxt("weights_ising_layer_3.csv",delimiter=",")
data4 = data4 / divider
maxes += [np.max(np.abs(data4))]
#plt.imshow(data3, cmap="hot")
#plt.colorbar()
#plt.show()
bluber =np.tanh(maxes)
plt.plot(np.tanh(maxes), np.tanh(maxes), "ro")

x = np.linspace(1,0,100)
y = np.tanh(np.arctanh(x))**2
f = lambda x: np.tanh(np.arctanh(x))**2
interpol = np.tanh(maxes)
params= np.polyfit(interpol, interpol, 1)
print(params)
plt.plot(x,y, "b")
plt.plot(x,params[0]*x + params[1], "b")

print(bluber)
#plot connections
line1 = []
print(line1)
for i in range(0, len(bluber)):
    plt.plot([bluber[i],bluber[i]], [bluber[i], f(bluber[i])], "r-.")
    plt.plot([bluber[i],f(bluber[i])], [f(bluber[i]), f(bluber[i])], "r-.")
plt.show()