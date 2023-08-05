# NOVO TESTE
# https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/reaction.inverse.html?highlight=component
"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np

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

# TODO cole as eqs aqui e veja se consegue rodar o reator...
# TODO ou melhor, como esse tá rodando, mantenha ele intacto de referência
# TODO e crie outro arquivo, mexa nele.

iterations = 1000

# kf = dde.Variable(0.05)
# D = dde.Variable(1.0)
kf = 0.05
D = 1.0

def pde(x, y):
    ca, cb = y[:, 0:1], y[:, 1:2]
    dca_t = dde.grad.jacobian(y, x, i=0, j=1)
    dcb_t = dde.grad.jacobian(y, x, i=1, j=1)
    return [dca_t, dcb_t]
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
ic1 = dde.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=0)
ic2 = dde.icbc.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=1)

# observe_x, Ca, Cb = gen_traindata()
# observe_y1 = dde.icbc.PointSetBC(observe_x, Ca, component=0)
# observe_y2 = dde.icbc.PointSetBC(observe_x, Cb, component=1)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_a, bc_b, ic1, ic2],# observe_y1, observe_y2],
    num_domain=2000,
    num_boundary=100,
    num_initial=100,
    # anchors=observe_x,
    num_test=50000,
)
net = dde.nn.FNN([2] + [20] * 3 + [2], "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=[kf, D])
# variable = dde.callbacks.VariableValue([kf, D], period=1000, filename="variables.dat")
# losshistory, train_state = model.train(iterations=80000, callbacks=[variable])
losshistory, train_state = model.train(iterations=iterations)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)


#------------------------------------------------
# # Roda esse
# # https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/burgers.html?highlight=GeometryXTime
# # E modifica só as besteiras pra ver se acho o problema

# # TODO veja como botar 2 variáveis de Entrada, e se buga. Já seria um começo.
# # COmo ele já tem, bote mais uma de saída.

# import deepxde as dde
# import numpy as np


# def gen_testdata():
#     data = np.load("../dataset/Burgers.npz")
#     t, x, exact = data["t"], data["x"], data["usol"].T
#     xx, tt = np.meshgrid(x, t)
#     X = np.vstack((np.ravel(xx), np.ravel(tt))).T
#     y = exact.flatten()[:, None]
#     return X, y


# def pde(x, y):
#     dy_x = dde.grad.jacobian(y, x, i=0, j=0)
#     dy_t = dde.grad.jacobian(y, x, i=0, j=1)
#     # dy_xx = dde.grad.hessian(y, x, i=0, j=0)

#     # O segredo parece estar nesse "component"
#     # Pelo que eu to entendendo nesse exemplo https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/reaction.inverse.html?highlight=component
#     # Usa o component pra tudo que não for o tempo, aí deixa i zerado mesmo.
#     # Ou isso ou quando for derivada de ordem 2?? Mas se fosse aqui seria por padrão tb
#     dy_xx = dde.grad.hessian(y, x, i=0, j=0, component=0)

#     # return dy_t + y * dy_x - 0.01 / np.pi * dy_xx
#     # FIXME essa mudança foi pra tornar multi output
#     # Mais de uma entrada parece ser ok
#     # é mais de uma entrada e saída ao mesmo tempo mds...
#     return [dy_t + y * dy_x - 0.01 / np.pi * dy_xx, 0]


# geom = dde.geometry.Interval(-1, 1)
# timedomain = dde.geometry.TimeDomain(0, 0.99)
# geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# # bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
# # A bc precisou ser convertida, tav dando o erro que era um int

# # Aparentemente não serviu pra nada criar esses extras aí de baixo:
# # TypeError: 'int' object is not subscriptable
# # File "C:\Python311\Lib\site-packages\deepxde\data\pde.py", line 162, in <listcomp>
# #     error_f = [fi[bcs_start[-1] :] for fi in f]

# bc0 = dde.icbc.DirichletBC(
#     geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=0
# )
# bc1 = dde.icbc.DirichletBC(
#     geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=1
# )
# ic0 = dde.icbc.IC(
#     geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial, component=0
# )
# ic1 = dde.icbc.IC(
#     geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial, component=1
# )

# data = dde.data.TimePDE(
#     geomtime,
#     pde,
#     # [bc, ic],
#     [bc0, bc1, ic0, ic1],
#     num_domain=2540,
#     num_boundary=80,
#     num_initial=160,
# )
# # net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
# # FIXME aqui transofrmei em multiple output
# # Erro: the component of y is missing. Pode ser pq tem que botar +1 ic? [bc, ic, ic],
# net = dde.nn.FNN([2] + [20] * 3 + [2], "tanh", "Glorot normal")
# model = dde.Model(data, net)

# model.compile("adam", lr=1e-3)
# # model.train(iterations=15000)
# model.train(iterations=1000)
# model.compile("L-BFGS")
# losshistory, train_state = model.train()
# dde.saveplot(losshistory, train_state, issave=False, isplot=True)

# X, y_true = gen_testdata()
# y_pred = model.predict(X)
# f = model.predict(X, operator=pde)
# print("Mean residual:", np.mean(np.absolute(f)))
# print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
# np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
