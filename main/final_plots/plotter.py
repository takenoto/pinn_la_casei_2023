# TODO
# faz aqui a o plot 3D das coisas 1 a 1 copiando da planilha
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


# TODO tentar outro plot pq esse não vai rolar
# de preferência um que aceite x, y, z, como pontos separados

def main():
    # Esse funciona realme nte como se fossem pares x, y, z
    # Copiado total de https://stackoverflow.com/questions/61791309/how-to-make-a-3d-plot-x-y-z-assigning-z-values-to-x-y-ordered-pairs
    # x , y = np.meshgrid(X,Y)
    # ax.plot_surface(x,y,np.array(Z).reshape(5,5))
    # plt.show()


    # return

    #----------------------------------------


    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # # Make data.
    # X = np.arange(-5, 5, 0.25)
    # Y = np.arange(-5, 5, 0.25)
    # X, Y = np.meshgrid(X, Y)
    # R = np.sqrt(X**2 + Y**2)
    # Z = np.sin(R)
    # # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')
    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                     linewidth=0, antialiased=False)
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    # print('X = ')
    # print(X)
    # print('\n\n')
    # print('Z = ')
    # print(Z)

    
    # -------------------------
    # EXEMPLO
    X = [1,2,3]
    Y = [4,5,6]
    # Z = [1,2,3,1,2,3] 
    # X HORIZONTAL Y VERTICAL MAS PARECE O CONTRÁRIO
    # 15 TÁ EM 3,6 
    # 14 TÁ EM 2, 6
    # 13 tá em 1, 6
    Z = [
        [7, 8, 9],
        [10, 11, 12],
        # [10, None, None] # None nem funcionou xisde
        [13, 14, 15]
        ]

    # --------------------------
    # DADOS INSERIDOS
    # X = [2,4,6,8,10] #
    X =   [2, 3, 4, 5, 6, 7, 8, 9, 10, 12] 
    # [ #layers
    #     2,
    #     4,
    #     6,
    #     8,
    #     10,
        
    # ]
    Y = [ #NL
        [4, 10, 22, 45, 70]
        # [4,],
        # [10,]
        # [22,],
        # [45,],
        # [70,],
        # [360,
        # 180,
        # 120,
        # 90,
        # 72],
        # [180,
        # 90,
        # 60,
        # 45,
        # 36],
        # [90,
        # 45,
        # 30,
        # 23,
        # 18],
    ]
    Y = []
    Y.append([4])
    Y.append([10])
    Y.append([22])
    Y.append([45])
    Y.append([70])

    Z = [
        # Agora o train time
        # 4 
        [
            28.871485299999996,
            35.7877414,
            None,#39.234360300000006,
            None,#41.57656329999999,
            43.940260499999994,
            47.93153140000001,
            51.521552499999984,
            57.6443974,
            61.7384217,
            72.08068559999998
        ],
        #10
        [
            30.417902499999997,
            36.589301999999996,
            46.527496799999994,
            45.10367179999999,
            50.01614759999998,
            56.54823759999999,
            62.1061234,
            69.4631101,
            78.57533590000003,
            90.25504590000003   
        ],
        #22
        [
        36.790539200000005,
        44.9004759,
        54.26644549999999,
        64.24002010000001,
        69.71282629999999,
        102.14496300000002,
        133.51168579999995,
        173.20102010000005,
        141.7936158,
        240.17598279999993
        ],
        #45
        [
            50.1717808,
            65.3616681,
            90.4347488,
            105.41792810000001,
            134.3705258,
            177.02162510000005,
            250.67946189999998,
            300.2285849,
            342.2662579,
            415.55508639999994
        ],
        #70
        [
            62.13429060000001,
            121.62549270000001,
            157.67520080000003,
            247.1600681,
            364.7569396,
            499.4327135000001,
            622.5035143,
            600.1904057000002,
            632.0614621000004,
            564.1440915999997
        ],


    #     # Primeiro vou fazer as loss...
    #     # 4
    #     [
    #         0.29779455065727234,
    #         0.07152257859706879,
    #         0.07185059040784836,
    #         0.3394952416419983,
    #         1.3021177053451538,
    #         47.69118118286133,
    #         25.102649688720703,
    #         1.302128553390503,
    #         1.4703738689422607,
    #         428.367431640625
    #     ],
    #     # 10
    #     [
    #     0.3149353265762329,
    #     0.33004000782966614,
    #     0.002160340314731002,
    #     0.000596635218244046,
    #     0.5261140465736389,
    #     0.01040086429566145,
    #     0.0007175933569669724,
    #     1.302128791809082,
    #     0.00028698862297460437,
    #     0.0009912136010825634,
    # ],
    # # 22
    #     [
    #     0.0027981088496744633,
    #     8.846916171023622e-05,
    #     4.8707708629081026e-05,
    #     25.611042022705078,
    #     36.508365631103516,
    #     41.26593017578125,
    #     0.001502437167800963,
    #     0.3273018002510071,
    #     0.2839580178260803,
    #     0.00019052563584409654
    # ],

    # # 45
    # [
    # 7.336396083701402e-05,
    # 0.00031183133251033723,
    # 0.31985417008399963,
    # 2.0138588297413662e-05,
    # 8.346907634404488e-06,
    # 1.837719901232049e-05,
    # 1.890239400381688e-05,
    # 2.511744969524443e-05,
    # 5.609665822703391e-05,
    # 2.13177663681563e-05
    # ],

    # # 70 
    # [
    #     0.31911540031433105,
    #     5.439895539893769e-05,
    #     5.739832340623252e-05,
    #     2.3945791326696053e-05,
    #     8.240695933636744e-06,
    #     0.5226672291755676,
    #     1.1360854841768742e-05,
    #     2.9079228625050746e-05,
    #     2.8323489459580742e-05,
    #     8.119521226035431e-05
    # ]



        # [2472.879395,
        # 5.00E+00,
        # 2.48E+03,
        # 1.47E+02,
        # 1.14E+02],
        # [813.625,
        # 3.56E-01,
        # 2.22E+02,
        # 1.79E+00,
        # 4.51E+02],
        # [820.1142578,
        # 5.45E+02,
        # 2.48E+03,
        # 547.4110107,
        # np.nan],
        ]
   

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    # ax.view_init(elev=30, azim=45)#, roll=15)
    ax.view_init(elev=25, azim=240)#, roll=15)
    # X, Y = np.meshgrid(X, Y)

    # TODO 1 - faz pra FB e cstr logo?
    # TODO faz um arquivo só com os X, Y e Z
    # TODO muda esquema de cores
    # TODO acha um ângulo melhor pelamor, e aumenta o tamanho da figsize
    # TODO fazer o 2D pra cada nº total fixo de neuronios
    # Aí posso avaliar efeito do nº camadas considerandou m numero total fixo mais facilmente
    # O gráfico seria beeem mais legível oxe
    cmap=None
    # cmap = 'terrain'
    cmap = 'summer'
    # cmap = 'autumn'
    # cmap = 'Paired'
    # cmap = 'Accent'
    # cmap = 'viridis_r'
    # Plot the surface.
    # Z = np.log(Z) 
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False)
    # ax.set_box_aspect([np.ptp(i) for i in (X, Y, Z)])
    fig.colorbar(surf, ax=ax)
    # surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False)
    # Inveret eixo x:
    # ax.set_xlim(np.max(X)*1.1, np.min(X)/1.1)
    # Inveret eixo y:
    # ax.set_ylim(np.max(Y)*1.1, np.min(Y)/1.1)

    x_label = 'layers'
    y_label = 'NL'
    # z_label = 'loss(test)'
    z_label = 'train time (s)'

    # if title:
    #     plt.title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if z_label:
        ax.set_zlabel(z_label)

    plt.show()


if __name__ == "__main__":
    main()