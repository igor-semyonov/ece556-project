import json
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def json_to_array(json_dict: dict[str, int | list[int]]):
    shape = json_dict["dim"]
    data = json_dict["data"]
    return np.array(data).reshape(shape)


def main():
    simulation = json.loads(Path("./simulation.json").read_text())

    memory = json_to_array(simulation["memory"])

    fig = plt.figure()
    ax = fig.add_subplot()
    line_cycler = cycle(["-", "--", "-.", ":"])
    for mem in memory.T:
        ax.plot(mem, linestyle=next(line_cycler))
    ax.legend(f"Neuron {i}" for i in range(memory.shape[1]))

    plt.get_current_fig_manager().window.showMaximized()
    plt.show()


if __name__ == "__main__":
    main()
