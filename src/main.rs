use ndarray as nd;
use serde::{Deserialize, Serialize};
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
        Self {
            u,
            beta,
            u_threshold,
            u_min,
        }
    }

    fn step(
        &self,
        spike_in: f64,
    ) -> (
        f64,
        f64,
    ) {
        let mut u_next =
            self.beta * self.u + spike_in;

        let mut spike_out = 0.0;

        if self.u >= self.u_threshold {
            u_next -= self.u_threshold;

            u_next = u_next.max(self.u_min);

            spike_out = 1.0;
        }

        let (u_next, spike_out) = (
            u_next, spike_out,
        );

        (
            u_next, spike_out,
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Layer {
    neurons: Vec<LeakyIntegrateAndFireNeuron>,
    internal_weights: nd::Array2<f64>, /* weights for neurons internal to this layer */
    memory: nd::Array2<f64>, /* [i, j] is potential of the jth neuron at time i */
    spikes_in: nd::Array2<f64>, /* spikes [i, j]
                              * is the spike
                              * amount received
                              * by the jth
                              * neuron at i */
    spikes_out: nd::Array2<f64>, /* spikes [i, j] is the spike amount produced by the jth neuron at i */
    n_steps: usize,
    i_step: usize,
}

impl Layer {
    fn new(
        neurons: Vec<LeakyIntegrateAndFireNeuron>,
        internal_weights: nd::Array2<f64>,
        spikes_in: nd::Array2<f64>,
        n_steps: usize,
    ) -> Self {
        let mut memory = nd::Array2::zeros([
            n_steps,
            neurons.len(),
        ]);

        let spikes_out = nd::Array2::zeros([
            n_steps,
            neurons.len(),
        ]);

        for idx in 0..neurons.len() {
            memory[[0, idx]] = neurons[0].u;
        }

        let i_step = 0usize;

        Self {
            neurons,
            internal_weights,
            memory,
            spikes_in,
            spikes_out,
            n_steps,
            i_step,
        }
    }

    fn n_neurons(&self) -> usize {
        self.neurons
            .len()
    }

    fn step(&mut self) {
        let current_spikes_in = self
            .spikes_in
            .slice(
                nd::s![
                    self.i_step,
                    ..
                ],
            )
            .into_owned();

        self.neurons
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

                    self.neurons[idx].u =
                        memory_next;

                    self.memory[[
                        self.i_step + 1,
                        idx,
                    ]] = memory_next;

                    for jdx in 0..self.n_neurons()
                    {
                        self.spikes_in[[
                            self.i_step + 1,
                            jdx,
                        ]] += spike_out_next
                            * self
                                .internal_weights
                                [[idx, jdx]];
                    }
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
    let neurons = (0..=2)
        .map(|_| {
            LeakyIntegrateAndFireNeuron::new(
                1.0, 0.9, 1.5, 0.2,
            )
        })
        .collect::<Vec<_>>();

    let internal_weights = nd::arr2(&[
        [
            0.0, 1.5, 1.0,
        ],
        [
            0.5, 0.0, 1.0,
        ],
        [
            0.5, 0.5, 0.0,
        ],
    ]);
    let mut spikes_in = nd::Array2::zeros([
        n_steps,
        neurons.len(),
    ]);

    spikes_in[[5, 0]] = 1.0;
    // spikes_in[[12, 1]] = 1.0;

    let mut layer_0 = Layer::new(
        neurons,
        internal_weights,
        spikes_in,
        n_steps,
    );

    layer_0.run();

    let json_string =
        serde_json::to_string(&layer_0).unwrap();

    fs::write(
        "./simulation.json",
        json_string,
    )
    .unwrap();
}
