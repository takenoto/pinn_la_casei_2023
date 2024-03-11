# python -m data.plot_caller.plot_caller

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

plt.style.use("./main/plotting/plot_styles.mplstyle")

from data.file_viewer.json_viewer import json_viewer

# Essa segunda parte do or é pra converter valores None pra erros altíssimos
noneVal = 9999999999900


def main():
    current_directory_path = os.getcwd()

    # Output directory
    output_folder = os.path.join(
        current_directory_path, "data", "plot_caller", "plot_caller_output"
    )
    # output_folder = os.path.join(
    #     current_directory_path, "results", "exported", "plot"
    # )

    # Input Directory
    input_folder = get_input_dir()

    jsons = []

    def callback(json):
        # Apenas pega os jsons que sejam wAutic
        if (
            json["pinn_model"]["params"]["solver_params"]["loss_weights"]["name"]
            == "autic-e2"
        ):
            jsons.append(json)

    json_viewer(callback=callback, folder_path=input_folder)

    # Ordering from max to min MAD error
    def sortJsonByMadError(jsons):
        # Stores a list of tuples with json + MAD
        jsons_and_MAD = []
        for json in jsons:
            jsons_and_MAD.append(
                (
                    json,
                    (json["error_MAD"]["X"] or noneVal)
                    + (json["error_MAD"]["P"] or noneVal)
                    + (json["error_MAD"]["S"] or noneVal)
                    + (json["error_MAD"]["V"] or 0),
                )
            )
        sortedJsons = sorted(
            jsons_and_MAD,
            key=lambda json_and_MAD: (json_and_MAD[-1]),
            reverse=True,
        )
        return sortedJsons

    jsons_and_MAD_sorted = sortJsonByMadError(jsons)
    for json_and_MAD in jsons_and_MAD_sorted:
        # MAD = json["error_MAD"]

        # totalMAD = (
        #     (MAD["X"] or noneVal)
        #     + (MAD["P"] or noneVal)
        #     + (MAD["S"] or noneVal)
        #     + (MAD["V"] or 0)
        # )
        # print(totalMAD)  # Tested and working fine
        json, MAD = json_and_MAD
        print(MAD)
        print(
            json["pinn_model"]["params"]["solver_params"]["nondim_scaler_input"]["name"]
            + " "
            + json["pinn_model"]["params"]["solver_params"]["nondim_scaler_output"][
                "name"
            ]
            + json["name"]
        )
    print("{} jsons analisados".format(len(jsons_and_MAD_sorted)))

    _create_ts_plot_heatmap(
        title="MAD Modelo Cinético: t_S",
        jsons_and_MAD=jsons_and_MAD_sorted,
        filename="MAD por ts REATOR CONTINUO CR",
    )

    # loss_test = json["best loss test"]
    # layer_size = json["solver_params"]["layer_size"]
    # NL = layer_size[1]
    # HL = len(layer_size) - 2

    # print("values")
    # print(values)

    # plot1_output = create_file_output_path(
    #     filename="Plot1 test.png", file_dir=output_folder
    # )

    # OK, parece que tudo que tenho que fazer é
    # 1) rodar todos os inputs desejados e agrupar eles de forma a facilitar a obtenção de dados
    # 2) criar estilo dos gráficos na plot 3D
    # Pronto. Então seria uma função pra cada coisa. Uma pra gerar os gráficos de loss, outra loss derivada,
    # etc. Preciso reler onde é que estão escritos cada gráfico e, se não tiver, fazer essa lista
    # de gráficos necessários.

    pass


def get_PINN_NL(pinn_json):
    return pinn_json["pinn_model"]["params"]["solver_params"]["layer_size"][1]


def get_PINN_HL(pinn_json):
    return len(pinn_json["pinn_model"]["params"]["solver_params"]["layer_size"]) - 2


def _create_ts_plot_heatmap(title, jsons_and_MAD, filename=None):
    """Ref: https://matplotlib.org/stable/gallery/images_contours_and_fields/multi_image.html#sphx-glr-gallery-images-contours-and-fields-multi-image-py"""

    # nplots deve ser um tuple dizendo como distribuir os gráficos?
    # Aí fazer uma escala só pra todos. Não sei como.
    # plots: {title, points}
    # title (título do gráfico)

    # This dict stores info like this: nondim_name: x, y, z
    nondim_groups = [
        "Lin-t1-1",
        "Lin-t2-F1",
        # "Lin-t2-F1x10",
        "Lin-t2-F1d10",
        "Lin-t3-F1d10",
        "Lin-t4-F1d10",
        "Lin-t5-F1d10",
        "Lin-t6-F1d10",
        "Lin-t7-F1d10",
        "Lin-t8-F1d10",
        "Lin-t9-F1d10",
    ]

    # NLs = [80, 45, 16]
    NLs = [80, 60, 35, 20, 10]
    HLs = [2, 4, 6, 8]

    n_rows = 2
    n_cols = 5

    assert n_rows * n_cols == len(nondim_groups), "Precisam ser iguais..."

    
    
    fig, axs = plt.subplots(n_cols, n_rows, figsize=(8, 12), dpi=400)
    fig.suptitle(title, y=1.0)
    fig.supxlabel("HL")
    fig.supylabel("NL")
    fig.tight_layout()

    images = []
    # Inicializado em -1 pra começar em 0 com a atualização logo no começo
    nd_number_order = -1
    # Útil para conseguir os valores max e min:
    all_data = []
    for nondim_name in nondim_groups:
        nd_number_order += 1
        data = []
        average_MAD = 0
        for NL in NLs:
            line_data = []
            for HL in HLs:
                # Encontra o PINN dentro da json list que atende a todos os critérios
                for json_and_MAD in jsons_and_MAD:
                    pinn, MAD = json_and_MAD
                    pinn_NL = get_PINN_NL(pinn)
                    pinn_HL = get_PINN_HL(pinn)
                    pinn_ND_name = pinn["pinn_model"]["params"]["solver_params"][
                        "nondim_scaler_input"
                    ]["name"]

                    if pinn_NL == NL and pinn_HL == HL and pinn_ND_name == nondim_name:
                        line_data.append(MAD)
                        all_data.append(MAD)
                        average_MAD += MAD

            data.append(line_data)
        average_MAD = average_MAD/(len(NLs)*len(HLs))
        print(f"{nondim_name} || len = {len(line_data)}")
        print(average_MAD)
        print(data)
        
        i = nd_number_order // (n_rows)
        j = nd_number_order % (
            n_rows
        )  # barras duplas é a syntaxe pra "floor division"
  
        ax = axs[i, j]
        images.append(
            ax.imshow(
                data,
                # Tipo de desfoque aplicado
                interpolation="gaussian",
                # Limites de fundo e topo
                # Fica desse jeito pq a imagem é construída é de baixo pra cima  e da esquerda pra direita
                extent=(HLs[0], HLs[-1], NLs[-1], NLs[0]),
                # Aspect faz com que não fique uma coisa meio estreita pq os valores são muito diferentes entre os eixos
                aspect="auto",
                # Color // RdBu  coolwarm
                cmap="RdBu_r",
            )
        )
        ax.set_xticks(HLs)

        ax.set_yticks(NLs)
        ax.label_outer()
        # É do 4 em diante pra excluir a parte "Lin-"
        ax.set_title(nondim_name[4:])

    # plt.pcolor(x, y, z, cmap="RdBu", vmin=minMAD, vmax=maxMAD)

    # Find the min and max of all colors for use in setting the color scale.
    # vmin = min(image.get_array().min() for image in images)
    vmin = np.min(all_data)
    # vmax = max(image.get_array().max() for image in images)
    vmax = np.max(all_data)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    # fig.colorbar(images[0], ax=axs, orientation="horizontal", fraction=0.1)
    fig.colorbar(
        images[0],
        ax=axs,
        orientation="vertical",
        ticks=np.linspace(vmin, vmax, 4),
        pad=0.04,
        fraction = 0.04,
    )

    # Make images respond to changes in the norm of other images (e.g. via the
    # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
    # recurse infinitely!
    def update(changed_image):
        for im in images:
            if (
                changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()
            ):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in images:
        im.callbacks.connect("changed", update)


    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    pass


def get_input_dir():
    current_directory_path = os.getcwd()
    # # Esses ** fazem com que busque dentro das subpastas
    # return os.path.join(
    #     current_directory_path, "data", "plot_caller", "plot_caller_input", "**"
    # )
    # # Se estiver dentro da própria pasta de input, sem subpastas, faça assim:
    # return os.path.join(
    #     current_directory_path,
    #     "data",
    #     "plot_caller",
    #     "plot_caller_input",
    # )
    # Pasta de resultados do CR com vazão maior (os que deram certo)
    return os.path.join(
        current_directory_path,
        "results",
        "exported",
        "reactor_altiok2006",
        "CR",
        "V0-5--Vmax-5--Fin-5E-1 f-inX0",
        "in_t-out_XPSV tr- 0-25pa Glorot uniform-Hammersley",
        "**",
    )


def create_file_output_path(filename, file_dir):
    return os.path.join(file_dir, filename)


if __name__ == "__main__":
    main()
