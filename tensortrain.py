import numpy as np
from tempnet import tempnet
import matplotlib.pyplot as plt
import tensorflow as tf

modelName = 'tempnet.model'
model = tempnet()

#savedEvalSet = np.load("evalSet.npy")
#evalImageSet = np.array([i[0] for i in savedEvalSet]).reshape(-1, inputWidth, inputHeight, 1)
#evalSolutionSet = np.array([i[1] for i in savedEvalSet])

inputSet =  [  [10],     [20],   [21],    [29],    [30],   [40]]
targetSet = [ [1,0,0], [1,0,0], [0,1,0], [0,1,0], [0,0,1], [0,0,1]]

#[Heizung,Nix,Klimaanlage]

model.fit({'input': inputSet}, {'targets': targetSet}, n_epoch=2,
            snapshot_step=500, show_metric=True, shuffle=True, run_id=modelName,batch_size=100000)

heizung = [0]*301
klima = [0]*301
temps = [0]*301
for temp in range (100,401):
    temp_arr = np.array([float(temp)/10])

    heizung[temp-100] = model.predict(temp_arr.reshape(-1, 1))[0][0]
    klima[temp-100] = model.predict(temp_arr.reshape(-1, 1))[0][2]
    temps[temp-100] = float(temp)/10

print(temps)
print(heizung)
model.save(modelName)

plt.plot(temps,heizung)
plt.plot(temps,klima)

plt.xlabel('Temperatur (in grad)')
plt.ylabel('Heizung %')
plt.title('About as simple as it gets, folks')
plt.grid(True)
plt.show()