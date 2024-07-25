#include "network.hpp"

#include "utils.hpp"

#include <random>

// Network functions

Network::Network(vec3d &weights,
        atomic3d &delta_weight,
        vec3d &delta_acc)
    : _weights(weights), _delta_weight(delta_weight), _delta_acc(delta_acc)
{
    for (uint l = 0; l < HIDDEN_LAYER_COUNT + 2; ++l) {
        _output.push_back(std::vector<double>(SHAPE[l], 0));
        if (l > 0) {
            // SHAPE[ l - 1 ] + 1 corresponds with added bias neuron

            _weights.push_back(create_matrix(SHAPE[l - 1] + 1, SHAPE[l], l));
            _delta_weight.push_back(create_matrix(SHAPE[l - 1] + 1, SHAPE[l]));
            _delta_acc.push_back(vec2d(SHAPE[l], std::vector<double>(SHAPE[l - 1] + 1, 0)));
        }
    }
    _gradient = _output;
    _potential = _output;
}

Network::Network(const Network &other)
    : _output(other._output),
      _gradient(other._gradient),
      _potential(other._potential),
      _weights(other._weights),
      _delta_weight(other._delta_weight),
      _delta_acc(other._delta_acc)
{
}

vec2d Network::create_matrix(int x, int y, int lo)
{
    // selu
    std::normal_distribution<double> distribution(0.0, 1.0 / (SHAPE[lo - 1] + 1));

    // relu
    //std::normal_distribution<double> distribution(0.0, 2.0 / (SHAPE[ lo - 1 ] + 1) );
    std::default_random_engine gen(0);
    vec2d vecvec;

    for (int i = 0; i < y; ++i) {
        std::vector<double> vec;
        for (int j = 0; j < x; ++j) {
            vec.push_back(distribution(gen));
        }
        vecvec.push_back(vec);
    }
    return vecvec;
}

atomic2d Network::create_matrix(int x, int y)
{
    atomic2d vecvec;

    for (int i = 0; i < y; ++i) {
        vecvec.emplace_back(std::vector<Atomdouble>(x, std::atomic<double>(0)));
    }
    return vecvec;
}

// DEBUG
void Network::print_3d(const std::string &name, const vec3d &vec) const
{
    std::cout << name << std::endl;
    for (const auto &i1 : vec) {
        for (const auto &i2 : i1) {
            for (const auto &i3 : i2) {
                std::cout << i3 << ", ";
            }
            std::cout << "   |   ";
        }
        std::cout << std::endl;
    }
}

// DEBUG
void Network::print_2d(std::string name, vec2d vec) const
{
    std::cout << name << std::endl;
    for (const auto &i1 : vec) {
        for (const auto &i2 : i1) {
            std::cout << i2 << ", ";
        }
        std::cout << std::endl;
    }
}

// ************* Setters ************* //

void Network::set_input(const std::vector<double> &input)
{
    for (uint i = 0; i < _output[0].size(); ++i) {
        _output[0][i] = double(input[i]);
    }
}

void Network::set_desired_res(const std::vector<int> &desired_res)
{
    _desired_result = desired_res;
}

// ************* Forward run ************* //

void Network::compute_layer_forward(size_t lo)
{
    assert(lo > 0);

    // Last layer
    if (lo == HIDDEN_LAYER_COUNT + 1) {
        for (uint n = 0; n < SHAPE[lo]; ++n) {
            auto pot = compute_potential(_weights[lo - 1][n], _output[lo - 1]);
            _potential[lo][n] = pot;
        }
        double bottom = exp_sum(_potential[lo]);
        for (uint n = 0; n < SHAPE[lo]; ++n) {
            _output[lo][n] = softmax(_potential[lo][n], bottom);
            assert(!std::isnan(_output[lo][n]));
        }
        return;
    }

    for (uint n = 0; n < SHAPE[lo]; ++n) {
        auto pot = compute_potential(_weights[lo - 1][n], _output[lo - 1]);
        _potential[lo][n] = pot;
        _output[lo][n] = get_output_func(lo, false)(pot);
        assert(!std::isnan(_output[lo][n]));
    }
}

// ************* Backpropagation ************* //

void Network::compute_gradient(int layer_index_o, int neuron_index)
{
    double res = 0;

    for (size_t r = 0; r < _gradient[layer_index_o + 1].size(); ++r) {                       // iterate over all neurons in upper layer
        res += _gradient[layer_index_o + 1][r]                                               // grad of prev layer
                * get_output_func(layer_index_o + 1, true)(_potential[layer_index_o + 1][r]) // get_output_func returns function based on lo index, if second arg is true, then returns derivative
                * _weights[layer_index_o][r][neuron_index + 1];                              // weight of connection between neuron_index and r
    }
    _gradient[layer_index_o][neuron_index] = res;
}

void Network::compute_layer_gradients(size_t layer_index_o)
{
    for (uint n = 0; n < SHAPE[layer_index_o]; ++n) {
        compute_gradient(layer_index_o, n);
    }
}

void Network::gradient_last_layer()
{
    const uint last_index = _gradient.size() - 1;

    for (size_t i = 0; i < _output.back().size(); ++i) {
        _gradient[last_index][i] = _desired_result[i] - _output.back()[i];
    }
}

// ************* Errors ************* //
void Network::compute_delta_error_last_layer()
{
    auto &vecvec = _weights.back();
    for (uint n = 0; n < vecvec.size(); ++n) {
        _delta_weight.back()[n][0] += _gradient.back()[n];
        for (uint w = 1; w < vecvec[n].size(); ++w) {
            _delta_weight.back()[n][w] += _gradient.back()[n] * _output[HIDDEN_LAYER_COUNT][w - 1];
        }
    }
}

void Network::compute_delta_error(size_t layer_index_w)
{
    auto lo = layer_index_w + 1; // conversion
    auto &layer_neurons = _weights[layer_index_w];

    for (uint n = 0; n < layer_neurons.size(); ++n) {
        auto out_f = get_output_func(lo, true)(_potential[lo][n]);
        auto val = _gradient[lo][n] * out_f;
        _delta_weight[layer_index_w][n][0] += val;
        assert(!std::isnan(_delta_weight[layer_index_w][n][0]._v));

        for (uint w = 1; w < layer_neurons[n].size(); ++w) {
            _delta_weight[layer_index_w][n][w] += _gradient[lo][n] * get_output_func(lo, true)(_potential[lo][n]) * _output[lo - 1][w - 1]; // w-1 because bias index conversion (to _output)
            assert(!std::isnan(_delta_weight[layer_index_w][n][w]._v));
        }
    }
}

double Network::get_squared_error()
{
    double res = 0;
    for (size_t i = 0; i < _output.back().size(); ++i) {
        res += (_output.back()[i] - _desired_result[i]) * (_output.back()[i] - _desired_result[i]);
    }
    return 0.5 * res;
}

double Network::get_cross_entropy_error()
{
    double res = 0;
    for (size_t i = 0; i < _output.back().size(); ++i) {
        res += _desired_result[i] * log(_output.back()[i]);
    }
    return -res;
}

// ************* Update/compute network ************* //

void Network::update_weights_and_reset()
{
    for (uint lw = 0; lw < _weights.size(); ++lw) {
        for (uint n = 0; n < _weights[lw].size(); ++n) {
            for (uint w = 0; w < _weights[lw][n].size(); ++w) {
                double change = -LEARNING_RATE * (-_delta_weight[lw][n][w]._v / double(BATCH_SIZE));
                _weights[lw][n][w] += change + (MOMENTUM * _delta_acc[lw][n][w]);

                _delta_acc[lw][n][w] = change + (MOMENTUM * _delta_acc[lw][n][w]);
                _delta_weight[lw][n][w] = 0; // set weight change to 0 for another epoch/batch

                // DEBUG
                assert(!std::isnan(_weights[lw][n][w]));
            }
        }
    }
}

void Network::forward_pass()
{
    //Forward pass
    // computes _output of each neuron
    for (size_t lo = 1; lo < HIDDEN_LAYER_COUNT + 2; lo++) {
        compute_layer_forward(lo);
    }
}

void Network::compute_network()
{
    forward_pass();

    // Backwardpass
    assert(HIDDEN_LAYER_COUNT >= 1);
    gradient_last_layer();
    for (uint lo = HIDDEN_LAYER_COUNT; lo > 0; --lo) {
        compute_layer_gradients(lo);
    }

    // Update errors
    for (size_t lw = 0; lw < HIDDEN_LAYER_COUNT + 1; lw++) {
        compute_delta_error(lw);
    }
}
