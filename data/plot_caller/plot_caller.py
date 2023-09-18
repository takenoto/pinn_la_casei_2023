# python -m data.plot_caller

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

    # TODO implementar callback
    # Edit this function to implement variations

    values = [
        # NL, HL, Loss_test (LoV)
    ]

    def callback(json):
        layer_size = json["solver_params"]["layer_size"]
        NL = layer_size[1]
        HL = len(layer_size) - 2
        loss_test = json["best loss test"]

        values.append((NL, HL, loss_test))

    json_viewer(callback=callback, folder_path=input_folder)

    plot1_output = create_file_output_path(
        filename="Plot1 test.png", file_dir=output_folder
    )

    pass


def get_input_dir():
    current_directory_path = os.getcwd()
    return os.path.join(
        current_directory_path, "data", "plot_caller", "plot_caller_input"
    )


def create_file_output_path(filename, file_dir):
    return os.path.join(file_dir, filename)


if __name__ == "__main__":
    main()
