#pragma once

#include <atomic>
#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <vector>

const uint HIDDEN_LAYER_COUNT = 3;
const uint INPUT_NEURONS = 784;
const uint OUTPUT_NEURONS = 10;
const uint SHAPE[HIDDEN_LAYER_COUNT + 2] = { INPUT_NEURONS, 150, 50, 40, OUTPUT_NEURONS };

const uint EPOCHS = 32;
const double LEARNING_RATE = 0.2;
const uint BATCH_SIZE = 512;

const double MOMENTUM = 0.5;

typedef std::vector<std::vector<std::vector<double>>> vec3d;
typedef std::vector<std::vector<double>> vec2d;
typedef std::vector<std::vector<int>> ivec2d;

typedef std::vector<uint> indexvec;

struct Atomdouble
{
    std::atomic<double> _v;

    Atomdouble()
        : _v()
    {
    }

    Atomdouble(const std::atomic<double> &v)
        : _v(v.load())
    {
    }

    Atomdouble(const Atomdouble &o)
        : _v(o._v.load())
    {
    }

    Atomdouble &operator=(const Atomdouble &o)
    {
        _v.store(o._v.load());
        return *this;
    }

    Atomdouble &operator=(const double &o)
    {
        _v.store(o);
        return *this;
    }

    Atomdouble &operator+=(const Atomdouble &r)
    {
        _v = _v + r._v;
        return *this;
    }

    Atomdouble &operator+=(const double &r)
    {
        _v = _v + r;
        return *this;
    }
};

typedef std::vector<std::vector<std::vector<Atomdouble>>> atomic3d;
typedef std::vector<std::vector<Atomdouble>> atomic2d;

class Network final
{
    vec2d _output;
    vec2d _gradient;
    vec2d _potential;

    vec3d &_weights;
    atomic3d &_delta_weight;
    vec3d &_delta_acc;

    std::vector<int> _desired_result;

  public:
    Network(vec3d &_weights,
            atomic3d &_delta_weight,
            vec3d &_delta_acc);

    Network(const Network &other);

    vec2d create_matrix(int x, int y, int lo);

    atomic2d create_matrix(int x, int y);

    vec3d &get_weights()
    {
        return _weights;
    }

    atomic3d &get_weight_change()
    {
        return _delta_weight;
    }

    vec2d &get_output()
    {
        return _output;
    }

    vec2d &get_potential()
    {
        return _potential;
    }

    vec2d &get_gradient()
    {
        return _gradient;
    }

    void print_3d(const std::string &name, const vec3d &vec) const;

    void print_2d(std::string name, vec2d vec) const;

    void set_input(const std::vector<double> &input);

    void set_desired_res(const std::vector<int> &desired_res);

    void compute_layer_forward(size_t lo);

    // Computes gradient for neuron given by indices l, n and updates _gradient vector
    void compute_gradient(int layer_index_o, int neuron_index);

    void compute_layer_gradients(size_t layer_index_o);

    void gradient_last_layer();

    void compute_delta_error_last_layer();

    void compute_delta_error(size_t layer_index_w);

    double get_squared_error();

    double get_cross_entropy_error();

    void update_weights_and_reset();

    void forward_pass();

    void compute_network();
};
