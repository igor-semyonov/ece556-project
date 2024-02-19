use ndarray as nd;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs;

#[derive(
    Debug, Clone, Serialize, Deserialize,
)]
struct LeakyIntegrateAndFireNeuron {
    u: f64, //membrane potential
    beta: f64,
    u_threshold: f64,
    u_min: f64,
}
impl LeakyIntegrateAndFireNeuron {
    fn new(
        u: f64,
        beta: f64,
        u_threshold: f64,
        u_min: f64,
    ) -> Self {
        Self { u, beta, u_threshold, u_min }
    }

    fn step(
        &self,
        spike_in: f64,
    ) -> (f64, f64) {
        let mut u_next =
            self.beta * self.u + spike_in;
        let mut spike_out = 0.0;
        if self.u >= self.u_threshold {
            u_next -= self.u_threshold;
            u_next = u_next.max(self.u_min);
            spike_out = 1.0;
        }
        let (u_next, spike_out) =
            (u_next, spike_out);
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
    spikes_in: nd::Array2<f64>, // spikes [i, j] is the spike amount received by the jth neuron at i
    spikes_out: nd::Array2<f64>, // spikes [i, j] is the spike amount produced by the jth neuron at i
    n_steps: usize,
    i_step: usize,
}
impl Simulation {
    fn new(
        network: Network,
        spikes_in: nd::Array2<f64>,
        n_steps: usize,
    ) -> Self {
        let mut memory =
            nd::Array2::zeros([
                n_steps,
                network.n_neurons(),
            ]);
        let spikes_out =
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
            spikes_in,
            spikes_out,
            n_steps,
            i_step,
        }
    }
    fn step(self: &mut Self) {
        let current_spikes_in = self
            .spikes_in
            .slice(nd::s![self.i_step, ..])
            .into_owned();
        self.network
            .neurons
            .clone()
            .into_iter()
            .zip(current_spikes_in)
            .enumerate()
            .for_each(
                |(
                    idx,
                    (neuron, spike_current),
                )| {
                    let (
                        memory_next,
                        spike_out_next,
                    ) = neuron
                        .step(spike_current);
                    self.network.neurons
                        [idx]
                        .u = memory_next;
                    self.memory[[
                        self.i_step + 1,
                        idx,
                    ]] = memory_next;
                    for jdx in 0..self
                        .network
                        .n_neurons()
                    {
                        self.spikes_in[[
                            self.i_step + 1,
                            jdx,
                        ]] += spike_out_next
                            * self
                                .network
                                .weights
                                [[idx, jdx]];
                    }
                    // let mut
                    // spikes_next_slice = self
                    //     .spikes
                    //     .slice_mut(nd::s![
                    //         self.i_step + 1,
                    //         ..
                    //     ]);
                    // let tmp =
                    //     &spikes_next_slice
                    //         * &self
                    //             .network
                    //             .weights
                    //             .slice(
                    //                 nd::s![
                    //                     idx,
                    //                     ..
                    //                 ],
                    //             )
                    //         * spike_next;
                    // spikes_next_slice +=
                    //     &tmp;
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
    let n_steps = 30usize;

    let neurons = (0..=1)
        .map(|_| {
            LeakyIntegrateAndFireNeuron::new(
                1.0, 0.9, 1.5, 0.2,
            )
        })
        .collect::<Vec<_>>();

    let weights =
        nd::arr2(&[[0.0, 1.5], [0.5, 0.0]]);
    let network =
        Network::new(neurons, weights);
    let mut spikes_in = nd::Array2::zeros([
        n_steps,
        network.n_neurons(),
    ]);

    spikes_in[[5, 0]] = 1.0;
    // spikes_in[[12, 1]] = 1.0;

    let mut simulation = Simulation::new(
        network, spikes_in, n_steps,
    );
    simulation.run();
    // dbg!(&simulation);

    let json_string =
        serde_json::to_string(&simulation)
            .unwrap();
    fs::write(
        "./simulation.json",
        &json_string,
    )
    .unwrap();
}
