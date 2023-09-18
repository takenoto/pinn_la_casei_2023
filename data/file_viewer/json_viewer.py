# python -m data.file_viewer.json_viewer
import json
import os
import glob
from typing import Callable

def json_viewer(callback:Callable[[dict], None], folder_path=None):
    """_summary_

    Args:
        folder_path (str): The path to the folder that contains all files to be opened.
        
        (function): Receives a [json] file that was read and should
        do the required processing of the info from the json. the file is
        closed after callback is called.
    """
    # ref: https://stackoverflow.com/questions/18262293/how-to-open-every-file-in-a-folder
    if folder_path:
        for filename in glob.glob(os.path.join(folder_path, "*.json")):
            print(filename)
        # for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), "r") as file:
                # ref: https://www.freecodecamp.org/news/loading-a-json-file-in-python-how-to-read-and-parse-json/
                parsed_json = json.load(file)
                callback(parsed_json)
                
    
    
    
def __test():
    # Ex: Layer 10x6: [1, 10, 10, 10, 10, 10, 10, 3]
    # Supondo que queremos só as layers 10x6 a 10x8:
    
    values = [
        #(NL, HL, Loss test)
    ]
    
    def callback_example(json):
        layer_size = json["solver_params"]["layer_size"]
        NL = layer_size[1]
        HL = len(layer_size)-2
        loss_test = json["best loss test"]
        
        values.append((NL, HL, loss_test))
    
    current_directory_path = os.getcwd()
    folder_path = os.path.join(
        current_directory_path, "data", "file_viewer", "sample"
    )
    json_viewer(callback=callback_example, folder_path=folder_path)
    
    print(values)     
        
    pass


if __name__ == "__main__":
    __test()