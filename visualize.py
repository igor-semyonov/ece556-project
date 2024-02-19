import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def json_to_array(json_dict: dict[str, int | list[int]]):
    shape = json_dict["dim"]
    data = json_dict['data']
    return np.array(data).reshape(shape)


def main():
    simulation = json.loads(Path("./simulation.json").read_text())

    memory = json_to_array(simulation["memory"])

    fig = plt.figure()
    ax = fig.add_subplot()
    for mem in memory.T:
        ax.plot(mem)

    plt.get_current_fig_manager().window.showMaximized()
    plt.show()


if __name__ == "__main__":
    main()
