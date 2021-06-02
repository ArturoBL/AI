import numpy as np

#Lista de valores de entrada
X = np.array([
       [0,0,0]
      ,[0,0,1]
      ,[0,1,0]
      ,[0,1,1]
      ,[1,0,0]
      ,[1,0,1]
      ,[1,1,0]
      ,[1,1,1]
     ])

#Lista de valores deseados
D = np.array(
    [  0
      ,0
      ,0
      ,0
      ,0
      ,0
      ,0
      ,1
     ])

#Lista inicial de pesos
W = np.array(
    [   0.5
      , 0.5
      , 0.5
     ]
)

n = 0.3             #Coeficiente de aprendizaje
t = 0.2             #Umbral
ep = 0              #Epoch

#funcion escalón
def factivacion(z):
    if z >= 0:
        return 1
    else:
        return 0

def test(x,w,t):
    z = x @ w - t
    y = factivacion(z)
    return y

def entrena(x,w,t,d,n):
    y = test(x,w,t)
    e = d - y
    dw = x * n * e
    dt = - n * e
    return w + dw , t + dt, e


errores = 1
ep = 0
while errores != 0:
    ex = 0,
    print("___________")
    errores = 0
    for i in range(0, len(X)):
        W,t, err = entrena(X[i],W,t,D[i],n)
        print("W: ",W)
        print("t: ",t)
        print("err: ", err)
        if ex == 0 and err != 0:
            errores = 1
    ep += 1
    print("ep: ",ep)
    print("errores epoch: ",ex)

print("Result:")
print("W: ",W)
print("T: ",t)

y = test(X[7],W,t)
print("Prueba: ",y)
#ww = W + 2
#print(ww)


