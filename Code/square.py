#analytical model for square function
from keras.models import Sequential,clone_model
from keras.layers import Dense,Activation
from scipy.linalg import block_diag
import numpy as np
import matplotlib.pyplot as plt

n = 4

phi_g = Sequential()

id_weights = [np.identity(1),np.array([0])]
l1_weights = [np.array([[1,1,1]]),np.array([0,-0.5,-2])]
l2_weights = [np.array([[2],[-4],[2]]),np.array([0])]
l3_weights = [np.array([[2,2,2],[-4,-4,-4],[2,2,2]]),np.array([0,-0.5,-2])]

"""

#function g
layer1 = Dense(3,input_dim=1)
phi_g.add(layer1)
layer1.set_weights(l1_weights)
#A,b = layer1.get_weights()
#print(A,b)

phi_g.add(Activation('relu'))

layer2 = Dense(1)
phi_g.add(layer2)
layer2.set_weights(l2_weights)


#phi_g.summary()

# affichage de g
A = np.linspace(0,1,100)
B = phi_g.predict(A,batch_size=1)

plt.plot(A,B)
plt.show()



#g_s

#identity with two layers
phi_id = Sequential()
phi_id.add(Dense(1,input_dim=1))
phi_id.layers[0].set_weights(id_weights)
phi_id.add(Dense(1))
phi_id.layers[0].set_weights(id_weights)

#g_s list
L_gs = [phi_id,phi_g]

L_gs[0].add(Dense(1))
L_gs[0].layers[-1].set_weights(id_weights)

#g_2
layer3 = Dense(3)
phi_gs = clone_model(L_gs[1])
phi_gs.set_weights(L_gs[1].get_weights())
phi_gs.pop()
phi_gs.add(layer3)
layer3.set_weights(l3_weights)
phi_gs.add(Activation('relu'))
phi_gs.add(Dense(1)) #same as layer2
phi_gs.layers[-1].set_weights(l2_weights)
#phi_gs.summary()
L_gs.append(clone_model(phi_gs))
L_gs[-1].set_weights(phi_gs.get_weights())

#g_s s in [3,n]
for i in range(3,n+1) :
    phi_gs = clone_model(L_gs[-1])
    for j in range(i) :# fill all models with identity layers to have the same number of layers
        L_gs[j].add(Dense(1))
        L_gs[j].layers[-1].set_weights(id_weights)
    phi_gs.set_weights(L_gs[-1].get_weights())
    phi_gs.pop()
    phi_gs.add(Dense(3)) #same as layer3
    phi_gs.layers[-1].set_weights(l3_weights)
    phi_gs.add(Activation('relu'))
    phi_gs.add(Dense(1)) #same as layer2
    phi_gs.layers[-1].set_weights(l2_weights)
    #phi_gs.summary()
    L_gs.append(clone_model(phi_gs))
    L_gs[-1].set_weights(phi_gs.get_weights())

"""

phi_square = Sequential()
phi_square.add(Dense(n+1,input_dim=1))
phi_square.layers[0].set_weights([np.ones((1,n+1)),np.zeros(n+1)])


weights_list = [id_weights]+[l1_weights]*n
mat_list = [w[0] for w in weights_list]
vect_list = [w[1] for w in weights_list]

A = block_diag(*mat_list)
b = np.concatenate(vect_list, axis=None)
phi_square.add(Dense(b.shape[0]))
phi_square.layers[-1].set_weights([A,b])
phi_square.add(Activation('relu'))

 
for i in range(n) :
    weights_list = [id_weights]*(i+1)+[l2_weights]+[l3_weights]*(n-i-1)
    
    mat_list = [w[0] for w in weights_list]
    vect_list = [w[1] for w in weights_list]

    A = block_diag(*mat_list)
    b = np.concatenate(vect_list, axis=None)
    phi_square.add(Dense(b.shape[0]))
    phi_square.layers[-1].set_weights([A,b])
    phi_square.add(Activation('relu'))


final_layer_weights = np.transpose(np.array([[1]+[-(1/2)**(2*s) for s in range(1,n+1)]]))
phi_square.add(Dense(1))
phi_square.layers[-1].set_weights([final_layer_weights,np.array([0])])

phi_square.summary()

#affichage
X = np.linspace(0,1,1000)
Y = [x**2 for x in X]

Y_n = phi_square.predict(X,batch_size=1)

Y_int = Y[:-1]
X_int = Y_n[:-1]
Y_int2 = Y[1:]
X_int2 = Y_n[1:]
erreur1 = sum([(x-y)**2 for x,y in zip(X_int,Y_int)])
erreur2 = sum([(x-y)**2 for x,y in zip(X_int2,Y_int2)])
erreur = (erreur1+erreur2)/(2*(len(X)-1)*(X[-1]-X[0]))
print("Erreur :",erreur)

E = np.abs(np.concatenate(Y_n)-Y)
print("max error :", np.amax(E))
print("mean error :", np.mean(E))

plt.plot(X,Y_n,'-x',label='y=fn(x)')
plt.plot(X,Y,label='y=x**2')
plt.legend()
plt.show()
