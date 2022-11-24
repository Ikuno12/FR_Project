import numpy as np
from copy import copy

cos = np.cos
sin = np.sin
pi = np.pi


def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.
    """
    # Escriba aqui la matriz de transformacion homogenea en funcion de los valores de d, theta, a, alpha
    return np.array([
        [cos(theta), -cos(alpha)*sin(theta),
         sin(alpha)*sin(theta), a*cos(theta)],
        [sin(theta), cos(alpha)*cos(theta), -
         cos(theta)*sin(alpha), a*sin(theta)],
        [0, sin(alpha), cos(alpha), d],
        [0, 0, 0, 1]
    ])


def fkine_robot(q):
    # Calcula la cinematica directa del robot robot dados sus valores articulares.
    # Longitudes (en metros)
    L = np.array([129.904, 0,       100, 0,   0,   0,   25.5])/1000
    a = np.array([0,       86.603,  0,   124, 124, 60,     0])/1000

    T0 = dh(L[0], 0,    a[0], 0)
    T1 = dh(L[1], q[0], a[0], np.deg2rad(30)+pi/2)
    T2 = dh(L[2], q[1], a[1], -pi/2)
    T3 = dh(L[3], q[2], a[2], 0)
    T4 = dh(L[4], q[3], a[3], 0)
    T5 = dh(L[5], q[4], a[4], pi/2)
    T6 = dh(L[6], q[5], a[5], 0)

    # Efector final con respecto a la base
    return T0.dot(T1).dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)


def jacobian_robot(q, delta=0.0001):
    # Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como

    # Crear una matriz 3x6
    J = np.zeros((3, 6))
    # Utilizar la funcion que calcula la cinematica directa, para encontrar x,y,z usando q
    xq = fkine_robot(q)

    # Iteracion para la derivada de cada columna
    for i in range(6):
        # Copiar la configuracion articular inicial
        dq = copy(q)

        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta

        # Transformacion homogenea luego del incremento (q+delta)
        xdq = fkine_robot(dq)

        # Aproximacion del Jacobiano de posicion usando diferencias finitas, para la articulacion i-esima
        J[0:3, i] = (xdq[0:3, 3] - xq[0:3, 3])/delta

    return J


def ikine_robot(xdes, q0):
    # Calcular la cinematica inversa de robot numericamente a partir de la configuracion articular inicial de q0.
    epsilon = 0.001
    max_iter = 1000
    delta = 0.00001

    q = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian_robot(q, delta)
        xq = fkine_robot(q)

        # error y calculo de q por pesudoinversa del jacobiano
        e = xdes - xq[0:3, 3]
        q = q + np.dot(np.linalg.pinv(J), e)

        # Termino por norma menor que epsilon
        if (np.linalg.norm(e) < epsilon):
            break

    return q


def ik_gradient_robot(xdes, q0):
    # Calcular la cinematica inversa de robot numericamente a partir de la configuracion articular inicial de q0. Emplea el metodo gradiente
    epsilon = 0.001
    max_iter = 1000
    delta = 0.00001

    q = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian_robot(q, delta)
        xq = fkine_robot(q)

        # error y calculo de q por jacobiano transpuesto
        e = xdes - xq[0:3, 3]
        q = q + np.dot(J.T, e)

        # Termino por norma menor que epsilon
        if (np.linalg.norm(e) < epsilon):
            break

    return q
