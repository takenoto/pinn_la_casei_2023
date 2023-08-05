# NOVO TESTE
# https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/reaction.inverse.html?highlight=component
"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
dde.config.set_random_seed(0)
"""
Component = 0 é claramente Ca e Component = 1 Cb
Mas eu não entendo a diferença disso pra tacar o i no index correspondente, sinceramente?????
Ele faz um de um jeito e no outro o i=0 pra ambos ???????
"""

# Movi o V pro começo e o NaN continuou na última posição da loss, mds...


# def gen_traindata():
#     data = np.load("../dataset/reaction.npz")
#     t, x, ca, cb = data["t"], data["x"], data["Ca"], data["Cb"]
#     X, T = np.meshgrid(x, t)
#     X = np.reshape(X, (-1, 1))
#     T = np.reshape(T, (-1, 1))
#     Ca = np.reshape(ca, (-1, 1))
#     Cb = np.reshape(cb, (-1, 1))
#     return np.hstack((X, T)), Ca, Cb

# TODO aliás agora que funcionou parcialmente, volte lá no X e bote ele pra ser = o P de saída
# Porque eu to derivando uma função de entrada em relação à outra, isso talvez não faça nenhum sentido!
# TODO criei mais uma saída (C) pra ver se consigo ao menos fazer rodar
# Deu perfeitamente certo...
# Onde to errando lá? 
# É melhor comentar o código e refazer logo as ics e bcs ?
# Talvez seja na fun_init e fun_bc?
# Esse aqui já tá funcionando, 2in 3out e são 9 valores printados no acompanhamento da loss
# acho que são 3 variáveis + 3 bcs + 3 ics esses 9
# Padrão. Nos que mudei só multipliquei o x[:,0:1] por zero
# [2.75e-03, 3.09e-02, 3.36e-03, 4.57e-01, 8.62e-01, 5.13e-01, 1.64e-01, 1.44e-01, 2.06e-01]
# Mudando bcs => bcs são os 3 do meio
# [2.75e-03, 3.09e-02, 3.36e-03, 8.21e-02, 1.22e+00, 7.13e-02, 1.64e-01, 1.44e-01, 2.06e-01]  
# Mudando ics => ics são os 3 finais
# [2.75e-03, 3.09e-02, 3.36e-03, 4.57e-01, 8.62e-01, 5.13e-01, 1.42e-01, 1.22e-01, 1.85e-01]    

iterations = 1000

# kf = dde.Variable(0.05)
# D = dde.Variable(1.0)
kf = 0.05
D = 1.0

def pde(x, y):
    ca, cb = y[:, 0:1], y[:, 1:2]
    dca_t = dde.grad.jacobian(y, x, i=0, j=1)
    dcb_t = dde.grad.jacobian(y, x, i=1, j=1)
    dcc_t = dde.grad.jacobian(y, x, i=2, j=1)
    return [dca_t, dcb_t, dcc_t]
    dca_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dcb_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    eq_a = dca_t - 1e-3 * D * dca_xx + kf * ca * cb ** 2
    eq_b = dcb_t - 1e-3 * D * dcb_xx + 2 * kf * ca * cb ** 2
    return [eq_a, eq_b]


def fun_bc(x):
    return 1 - x[:, 0:1]


def fun_init(x):
    return np.exp(-20 * x[:, 0:1])


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 10)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc_a = dde.icbc.DirichletBC(
    geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=0
)
bc_b = dde.icbc.DirichletBC(
    geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=1
)
bc_c = dde.icbc.DirichletBC(
    geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=2
)
ic1 = dde.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=0)
ic2 = dde.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=1)
ic3 = dde.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=2)

# observe_x, Ca, Cb = gen_traindata()
# observe_y1 = dde.icbc.PointSetBC(observe_x, Ca, component=0)
# observe_y2 = dde.icbc.PointSetBC(observe_x, Cb, component=1)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_a, bc_b, bc_c, ic1, ic2, ic3],# observe_y1, observe_y2],
    num_domain=2000,
    num_boundary=100,
    num_initial=100,
    # anchors=observe_x,
    num_test=50000,
)
net = dde.nn.FNN([2] + [20] * 3 + [3], "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=[kf, D])
# variable = dde.callbacks.VariableValue([kf, D], period=1000, filename="variables.dat")
# losshistory, train_state = model.train(iterations=80000, callbacks=[variable])
losshistory, train_state = model.train(iterations=iterations)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

