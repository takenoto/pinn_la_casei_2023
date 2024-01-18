# python -m data.plot_caller.plot_caller

import os


from data.file_viewer.json_viewer import json_viewer


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
        jsons.append(json)

    json_viewer(callback=callback, folder_path=input_folder)

    # Ordering from max to min MAD error
    def sortJsonByMadError(jsons):
        sortedJsons = sorted(
            jsons,
            key=lambda json: json["error_MAD"]["X"]
            + json["error_MAD"]["P"]
            + json["error_MAD"]["S"]
            + json["error_MAD"]["V"],
            reverse=True,
        )
        return sortedJsons

    sortedJsons = sortJsonByMadError(jsons)
    for json in sortedJsons:
        MAD = json["error_MAD"]
        totalMAD = MAD["X"] + MAD["P"] + MAD["S"] + MAD["V"]
        print(totalMAD)  # Tested and working fine
        print(
            json["pinn_model"]["params"]["solver_params"]["nondim_scaler_input"]["name"]
            + " "
            + json["pinn_model"]["params"]["solver_params"]["nondim_scaler_output"][
                "name"
            ]
            + json["name"]
        )
    print("{} jsons analisados".format(len(sortedJsons)))

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
