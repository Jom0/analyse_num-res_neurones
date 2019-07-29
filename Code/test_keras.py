#first tests
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
"""
#identity in R^n
n = 1

model = Sequential()
model.add(Dense(n,input_dim=n))
model.summary()

model.compile(optimizer='sgd',loss='mean_squared_error')

data = np.array([np.random.random([n]) for i in range(100)])
data_t = data[:70]
labels_t = data[:70]
data_v = data[70:]
labels_v = data[70:]

history = model.fit(data_t,labels_t,batch_size = 20,epochs = 10,validation_data = (data_v,labels_v))
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(loss_values)+1)
"""
"""
plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
"""
"""
linsp = np.linspace(0,1,5)
#test_data = np.array([[i,j] for i in linsp for j in linsp]) ##n = 2
test_data = np.array([i for i in linsp])
print("evaluate")
print(model.evaluate(test_data,test_data,batch_size = 5))
print("predict")
print(test_data)
pred = model.predict(test_data,batch_size = 1)
print(pred)

A,b = model.get_weights()
print("poids A, b =", A,b)

"#accurate model for identity
truemodel = Sequential()
truemodel.add(Dense(n,input_dim=n))
truemodel.set_weights([np.identity(n),np.zeros_like(b)])
"""
"""
#graph for n = 1
plt.plot(linsp,linsp,label = 'y=x')
for x in data :
    plt.plot(x,x,'-x')
plt.plot(linsp,pred,label = 'prediction')
plt.legend()
plt.show()

"""
eps = 0.3
linsp = np.linspace(-2,2,1000)
y = np.zeros(1000)
print(-1*eps)
for i in range(1000):
    if linsp[i]>0 :
        y[i] = 1
    elif linsp[i]>(-1*eps) :
        y[i] = 1/eps*linsp[i]+1

plt.plot(linsp,y)
plt.show()
