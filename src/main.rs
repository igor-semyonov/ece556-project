use ndarray as nd;
use serde::Serialize;
use serde_json;
use std::fs;

struct Network {
    neurons: Vec<LeakyIntegrateAndFireNeuron>,
    interaction_matrix: nd::Array2<f64>, // [i,j] is the weight for spikes from neuron i to j
    inputs: nd::Array3<f64>,
}

struct LeakyIntegrateAndFireNeuron {
    u: f64, //membrane potential
    beta: f64,
    u_threshold: f64,
}

impl LeakyIntegrateAndFireNeuron {
    fn new(
        u: f64,
        beta: f64,
        u_threshold: f64,
    ) -> Self {
        Self { u, beta, u_threshold }
    }

    fn step(&mut self, spike_in: f64) -> (f64, f64) {
        let mut u_next = self.beta * self.u + spike_in;
        let mut spike_out = 0.0;
        if u_next >= self.u_threshold {
            u_next -= self.u_threshold;
            spike_out = 1.0;
        }
        let (u_next, spike_out) = (u_next, spike_out);
        self.u = u_next;
        (u_next, spike_out)
    }
}

fn main() {
    let n_steps = 200usize;
    let mut memory: Vec<f64> =
        Vec::with_capacity(n_steps);
    let mut spikes_in =
        (0..n_steps).map(|_| 0.0).collect::<Vec<_>>();
    spikes_in[5] = 1.9;
    spikes_in[9] = 1.9;
    spikes_in[20] = 1.9;
    let mut spikes_out: Vec<f64> =
        Vec::with_capacity(n_steps);
    let mut neuron = LeakyIntegrateAndFireNeuron::new(
        1.0, 0.98, 1.5,
    );

    for i_step in 0..n_steps {
        let (mem, spike_out) =
            neuron.step(spikes_in[i_step]);
        memory.push(mem);
        spikes_out.push(spike_out);
    }

    let json_str = serde_json::to_string(&(memory, spikes_in, spikes_out)).expect("Unable to convert simulation results to json!");

    fs::write("./sim.json", json_str)
        .expect("Unable to write json!");
}
