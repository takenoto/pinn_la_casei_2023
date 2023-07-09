# TODO
# faz aqui a o plot 3D das coisas 1 a 1 copiando da planilha
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# de preferência um que aceite x, y, z, como pontos separados

def main():
    # Esse funciona realme nte como se fossem pares x, y, z
    # Copiado total de https://stackoverflow.com/questions/61791309/how-to-make-a-3d-plot-x-y-z-assigning-z-values-to-x-y-ordered-pairs
    # x , y = np.meshgrid(X,Y)
    # ax.plot_surface(x,y,np.array(Z).reshape(5,5))
    # plt.show()


    # return

    #----------------------------------------
    
    data_cstr_mbs_test13 = [
    #mb20 - layers 3, 6, 12
    [1166.5561981, 3.1299548149108887, 2.7957827946269975, 53.55543081465584, 420.3809034782482, 7.605647636787937, 484.33776472431896],
    [2524.2211946999996, 14.772445678710938, 1.6886525036201216, 4.528190475946858, 6.60516229866271, 0.04662014584117578, 12.868625424070865],
    [2926.8532747 , 0.0038529864978045225, 6.6326111599908595, 28.401439165107625, 36.02103647094642, 0.07938752781939089, 71.1344743238643],
    #mb100 - layers 3, 6, 12
    [3556.9480389, 4.4838361740112305, 26.740501292060454, 142.37541974289442, 2.647826422224827, 8.534398902790592, 180.2981463599703],
    [1552.9820059000012 , 2.1585257053375244, 17.355863617371163, 84.21571877105255, 2.5070156031496826, 8.998566610949085, 113.07716460252247],
    [3314.412362099998 , 8.489331245422363, 4.562552303583063, 29.8272801866454, 24.554650775376214, 9.826105737027373, 68.77058900263205],
    #mb1000 - layers 3, 6, 12
    [526.9628766000001 , 1.8890924453735352, 3.0552060269481314, 25.410719878755224, 89.1446426878206, 8.949064423726842, 126.5596330172508],
    [1260.1264307, 2.455801010131836, 2.960690975757882, 29.945365036807676, 148.62777845857866, 8.58959108872895, 190.12342555987317],
    [2255.7795591 , 0.003942422568798065, 0.9327164870953498, 5.9746177333216215, 5.535709336109929, 0.07843515807144864, 12.521478714598349]
    ]
    
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
    X = [3, 6, 12]
    # [ #layers
    #     2,
    #     4,
    #     6,
    #     8,
    #     10,
    Y = []
    #Y no geral são os neurônios
    # Y.append([22])
    # Y.append([45])
    # Y.append([70])
    # Y.append([90])
    # Y.append([130])
    # Y.append([200])
    Y.append([20])
    Y.append([100])
    Y.append([1000])


    data = data_cstr_mbs_test13
    titles = ['TRT(s)', 'loss', 'MAD_X', 'MAD_P', 'MAD_S', 'MAD_V', 'MAD_total']
    for t in range(len(titles)):
        title = titles[t]
        Z = []
        Z_scatter = []
        nl_list_size = 3 #3 # a cada quantos vai "quebrar" os dados em lista
        index_of_variable = t # index da variável de interesse. Train time = 0
        # como são 4 valores de NL neurônios e separei por NL neurônio é a cada 4
        # inputs_Errors_fb19
        for i in range(len(data)):
            if i%nl_list_size == 0:
                # Cria uma lista vazia
                Z.append([])
            # val = np.log(inputs_Errors_fb19[i][index_of_variable])
            val = data[i][index_of_variable]
            if False and val is not None:
                val = np.log(val)
            # O primeiro item é o train time... vamos testar
            Z[-1].append(val)
    

        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)


        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")
        # ax.view_init(elev=30, azim=45)#, roll=15)
        ax.view_init(elev=25, azim=240)#, roll=15)
        ax.view_init(elev=45, azim=120)#, roll=15)
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
        # ax.set_box_aspect([np.ptp(i) for i in (X, Y, Z)])
        # TODO veja como que faz o scatter
        # acho que vai ser mais fácil zzzzzzzzzzzzzzzzzzzzzzzzzz
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False)
        fig.colorbar(surf, ax=ax)
        # fig.scatter(X, Y, Z) 
        # surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False)
        # Inveret eixo x:
        # ax.set_xlim(np.max(X)*1.1, np.min(X)/1.1)
        # Inveret eixo y:
        # ax.set_ylim(np.max(Y)*1.1, np.min(Y)/1.1)

        x_label = 'HL'
        y_label = 'mb'
        # z_label = 'loss(test)'
        z_label = title

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