use ndarray as nd;
use serde::Serialize;
use serde_json;
use std::fs;

struct Network {
    neurons: Vec<LeakyIntegrateAndFireNeuron>,
    weights: nd::Array2<f64>, // [i,j] is the weight for spikes from neuron i to j
}

impl Network {
    fn new(
        neurons: Vec<LeakyIntegrateAndFireNeuron>,
        weights: nd::Array2<f64>,
    ) -> Self {
        Network { neurons, weights }
    }
    fn n_neurons(&self) -> usize {
        self.neurons.len()
    }
}

struct ForwardPass {
    network: Network,
    memory: nd::Array2<f64>, // [i, j] is potential of the jth neuron at time i
    spikes: nd::Array2<f64>, // spikes [i, j] is the spike amount received by the jth neuron at i
    n_steps: usize,
    i_step: usize,
}
impl ForwardPass {
    fn new(
        network: Network,
        spikes: nd::Array2<f64>,
        n_steps: usize,
    ) -> Self {
        let memory = nd::Array2::zeros([
            n_steps,
            network.n_neurons(),
        ]);
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
                |(idx, (neuron, spike_current))| {
                    let (memory_next, spike_next) =
                        neuron.step(*spike_current);
                    self.memory
                        [[self.i_step + 1, idx]] =
                        memory_next;
                    let mut spikes_next_slice =
                        self.spikes.slice(nd::s![
                            self.i_step + 1,
                            ..
                        ]);
                    let weights_slice = &self
                            .network
                            .weights
                            .slice(nd::s![idx, ..]);
                    spikes_next_slice +=
                        &spikes_next_slice * &weights_slice;
                },
            );
    }
}

#[derive(Clone)]
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

    fn step(&self, spike_in: f64) -> (f64, f64) {
        let mut u_next = self.beta * self.u + spike_in;
        let mut spike_out = 0.0;
        if u_next >= self.u_threshold {
            u_next -= self.u_threshold;
            spike_out = 1.0;
        }
        let (u_next, spike_out) = (u_next, spike_out);
        (u_next, spike_out)
    }
}

fn main() {
    let n_steps = 200usize;
    let mut spikes_in =
        (0..n_steps).map(|_| 0.0).collect::<Vec<_>>();
    spikes_in[5] = 1.9;
    spikes_in[9] = 1.9;
    spikes_in[20] = 1.9;

    let neurons = (0..=1)
        .map(|_| {
            LeakyIntegrateAndFireNeuron::new(
                1.0, 0.98, 1.5,
            )
        })
        .collect::<Vec<_>>();
    let weights = nd::arr2(&[[1.0, 1.0], [1.0, 1.0]]);
    let mut network = Network::new(neurons, weights);
    let mut spikes = nd::Array2::zeros([
        n_steps,
        network.n_neurons(),
    ]);

    let forward_pass =
        ForwardPass::new(network, spikes, n_steps);
    dbg!(forward_pass.memory);
}
