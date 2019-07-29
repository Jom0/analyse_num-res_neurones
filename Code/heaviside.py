#heaviside with sigmoid
from keras.models import Sequential,clone_model
from keras.layers import Dense,Activation
import numpy as np
import matplotlib.pyplot as plt

phi_h = Sequential()

phi_h.add(Dense(1,activation='sigmoid',input_dim=1))

phi_h.layers[0].set_weights([np.array([[1000]]),np.array([0])])


linsp = np.linspace(-1,1,100)
pred = phi_h.predict(linsp,batch_size = 1)

plt.plot(linsp,pred,'x',label = 'prediction')

plt.legend()
plt.show()
