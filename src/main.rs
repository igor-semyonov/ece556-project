use ndarray as nd;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    fn step(
        &mut self,
        spike_in: f64,
    ) -> (f64, f64) {
        let mut u_next =
            self.beta * self.u + spike_in;
        let mut spike_out = 0.0;
        if u_next >= self.u_threshold {
            u_next -= self.u_threshold;
            spike_out = 1.0;
        }
        let (u_next, spike_out) =
            (u_next, spike_out);
        self.u = u_next;
        (u_next, spike_out)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Network {
    neurons:
        Vec<LeakyIntegrateAndFireNeuron>,
    weights: nd::Array2<f64>, // [i,j] is the weight for spikes from neuron i to j
}
impl Network {
    fn new(
        neurons: Vec<
            LeakyIntegrateAndFireNeuron,
        >,
        weights: nd::Array2<f64>,
    ) -> Self {
        Network { neurons, weights }
    }
    fn n_neurons(&self) -> usize {
        self.neurons.len()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Simulation {
    network: Network,
    memory: nd::Array2<f64>, // [i, j] is potential of the jth neuron at time i
    spikes: nd::Array2<f64>, // spikes [i, j] is the spike amount received by the jth neuron at i
    n_steps: usize,
    i_step: usize,
}
impl Simulation {
    fn new(
        network: Network,
        spikes: nd::Array2<f64>,
        n_steps: usize,
    ) -> Self {
        let mut memory =
            nd::Array2::zeros([
                n_steps,
                network.n_neurons(),
            ]);
        for idx in 0..network.n_neurons() {
            memory[[0, idx]] =
                network.neurons[0].u;
        }
        let i_step = 0usize;
        Self {
            network,
            memory,
            spikes,
            n_steps,
            i_step,
        }
    }
    fn step(self: &mut Self) {
        let spikes_current = &self
            .spikes
            .slice(nd::s![self.i_step, ..])
            .dot(&self.network.weights);
        self.network
            .neurons
            .clone()
            .into_iter()
            .zip(spikes_current)
            .enumerate()
            .for_each(
                |(
                    idx,
                    (mut neuron, spike_current),
                )| {
                    let (
                        memory_next,
                        spike_next,
                    ) = neuron.step(
                        *spike_current,
                    );
                    self.memory[[
                        self.i_step + 1,
                        idx,
                    ]] = memory_next;
                    let mut
                    spikes_next_slice = self
                        .spikes
                        .slice_mut(nd::s![
                            self.i_step + 1,
                            ..
                        ]);
                    let tmp =
                        &spikes_next_slice
                            * &self
                                .network
                                .weights
                                .slice(
                                    nd::s![
                                        idx,
                                        ..
                                    ],
                                )
                            * spike_next;
                    spikes_next_slice +=
                        &tmp;
                },
            );
    }
    fn run(&mut self) {
        for step in 0..self.n_steps - 1 {
            self.i_step = step;
            self.step();
        }
    }
}

fn main() {
    let n_steps = 4usize;

    let neurons = (0..=1)
        .map(|_| {
            LeakyIntegrateAndFireNeuron::new(
                1.0, 0.98, 1.5,
            )
        })
        .collect::<Vec<_>>();
    
    let weights =
        nd::arr2(&[[1.0, 1.0], [1.0, 1.0]]);
    let network =
        Network::new(neurons, weights);
    let spikes = nd::Array2::zeros([
        n_steps,
        network.n_neurons(),
    ]);

    let mut simulation = Simulation::new(
        network, spikes, n_steps,
    );
    simulation.run();
    dbg!(&simulation);

    let json_string =
        serde_json::to_string(&simulation)
            .unwrap();
    fs::write(
        "./simulation.json",
        &json_string,
    )
    .unwrap();
}
