from keras.models import Sequential,clone_model
from keras.layers import Dense,Activation
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

phi_id = Sequential()
#same strucutre as x**2 for n=4
phi_id.add(Dense(5,activation='relu',input_dim=1))
phi_id.add(Dense(13,activation='relu'))
phi_id.add(Dense(11,activation='relu'))
phi_id.add(Dense(9,activation='relu'))
phi_id.add(Dense(7,activation='relu'))
phi_id.add(Dense(5,activation='relu'))
phi_id.add(Dense(1))

"""
#accurate model
phi_id.layers[0].set_weights([np.array([[1,-1]]),np.zeros(2)])
phi_id.layers[1].set_weights([np.array([[1],[-1]]),np.zeros(1)])
"""

"""
#same structure as accurate model
phi_id.add(Dense(2,activation='relu',input_dim=1))
phi_id.add(Dense(1))
"""
learning_rate = 0.001#default =0.01
decay_rate = 49/ 40 #default = 0
momentum = 0.99 #default = 0
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
phi_id.compile(optimizer='RMSprop',loss='mse',metrics=['mae'])

data = np.array([np.random.random()*2-1 for i in range(1000)])
data_t = data[:700]
target_t = data[:700]
data_v = data[700:]
target_v = data[700:]

history = phi_id.fit(data_t,target_t,batch_size = 20,epochs = 40, validation_data = (data_v,target_v))
history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
mae_values = history_dict['mean_absolute_error']
val_mae_values = history_dict['val_mean_absolute_error']

epochs = range(1,len(loss_values)+1)

fig = plt.figure()

ax1 = fig.add_subplot(311)
plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
ax2 = fig.add_subplot(312)
plt.plot(epochs,mae_values,'bo',label='Training mae')
plt.plot(epochs,val_mae_values,'b',label='Validation mae')
plt.xlabel('Epochs')
plt.ylabel('Mae')
plt.legend()
ax3 = fig.add_subplot(313)
linsp = np.linspace(-1,1,1000)
pred = phi_id.predict(linsp,batch_size = 1)
y_target = linsp
plt.plot(linsp,y_target,label = 'y=x')
plt.plot(linsp,pred,label = 'prediction')

ax1.title.set_text('Training and validation loss')
ax2.title.set_text('Training and validation mae')
ax3.title.set_text('phi_square')

Y_int = y_target[:-1]
X_int = pred[:-1]
Y_int2 = y_target[1:]
X_int2 = pred[1:]
erreur1 = sum([(x-y)**2 for x,y in zip(X_int,Y_int)])
erreur2 = sum([(x-y)**2 for x,y in zip(X_int2,Y_int2)])
erreur = (erreur1+erreur2)/(2*(len(linsp)-1)*(linsp[-1]-linsp[0]))
print("Erreur :",erreur)
E = np.abs(np.concatenate(pred)-y_target)
print("max error :", np.amax(E))
print("mean error :", np.mean(E))
plt.legend()

plt.show()
"""
weights = phi_id.get_weights()

print("weights:",weights)
"""
