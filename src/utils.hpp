#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

const double STEEPNESS = 1;

double compute_potential(const std::vector<double> &neuron_weights, const std::vector<double> &incoming_output)
{
    // neuron_weights have bias as leftmost item
    assert(neuron_weights.size() == incoming_output.size() + 1);

    double res = neuron_weights[0]; // bias (fictionally multiplied by 1)
    for (size_t i = 0; i < incoming_output.size(); ++i) {
        res += neuron_weights[i + 1] * incoming_output[i];
    }
    return res;
}

double exp_sum(const std::vector<double> &potentials)
{
    double res = 0;
    for (auto potential : potentials) {
        res += exp(potential);
    }
    return res;
}

double softmax(double potential, double bottom)
{
    assert(bottom != 0);
    return (exp(potential)) / bottom;
}

double dsoftmax(double)
{
    return 1;
}

double unit_step(double x)
{
    return (x >= 0) ? double(1) : (double) 0;
}
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x * STEEPNESS));
} // -x should be multiplied by lambda (steepness paramater)
double dsigmoid(double x)
{
    return sigmoid(x) * (1.0 - sigmoid(x));
} // derivation of sigmoid

double relu(double x)
{
    return (x < 0) ? 0.0 : x;
}
double drelu(double x)
{
    return (x < 0) ? 0.0 : 1.0;
}

double selu(double x)
{
    return (x < 0.0) ? (1.05 * 1.673 * (std::exp(x) - 1)) : 1.05 * x;
}
double dselu(double x)
{
    return (x < 0.0) ? (1.05 * 1.673 * std::exp(x)) : 1.05;
}

std::function<double(double)> get_output_func(int lo, bool derivation)
{
    // return function based on layer
    if (lo == HIDDEN_LAYER_COUNT + 1) {
        // last layer
        if (derivation) {
            return dsoftmax;
        }
        assert(false);
    }

    // hidden layer
    if (derivation) {
        return dselu;
    }
    return selu;
}
