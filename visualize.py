import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    sim_data = json.loads(Path("./sim.json").read_text())
    memory, spikes_out, spikes_in = sim_data
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(memory)

    plt.get_current_fig_manager().window.showMaximized()
    plt.show()


if __name__ == "__main__":
    main()
