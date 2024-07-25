#include "network.hpp"

#include <algorithm> // std::random_shuffle
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>   // std::stringstream
#include <stdexcept> // std::runtime_error
#include <vector>

// openmp parallel cykly

const std::string CSV_TRAIN_INPUT = "./data/fashion_mnist_train_vectors.csv";
const std::string CSV_TRAIN_RESULT = "./data/fashion_mnist_train_labels.csv";

const std::string CSV_TEST_INPUT = "./data/fashion_mnist_test_vectors.csv";
const std::string CSV_TEST_RESULT = "./data/fashion_mnist_test_labels.csv";

// total is 60000
//const size_t TRAIN_SIZE = 50000;
const size_t VALIDATION_SIZE = 10000;
const uint THREADS = 32; //64;

// read file
std::vector<int> create_ouput_enc(int val)
{
    std::vector<int> vec(10, 0);
    vec[val] = 1;
    return vec;
}

ivec2d read_csv_output(const std::string &filename)
{
    ivec2d result;
    std::ifstream myFile(filename);
    if (!myFile.is_open())
        throw std::runtime_error("Could not open file");

    std::string line;
    int val;

    while (std::getline(myFile, line)) {
        // Create a stringstream of the current line
        std::stringstream ss(line);
        std::vector<int> oneline;

        // Extract each integer
        while (ss >> val) {
            // encodes output (2) -> (0,0,1,0,0,0,0,0,0,0)
            oneline = create_ouput_enc(val);

            // If the next token is a comma, ignore it and move on
            if (ss.peek() == ',')
                ss.ignore();
        }

        result.push_back(oneline);
    }

    // Close file
    myFile.close();
    return result;
}

vec2d read_csv_input(const std::string &filename)
{
    //
    vec2d result;
    std::ifstream myFile(filename);
    if (!myFile.is_open())
        throw std::runtime_error("Could not open file");

    std::string line;
    double val;

    while (std::getline(myFile, line)) {
        // Create a stringstream of the current line
        std::stringstream ss(line);
        std::vector<double> oneline;

        // Extract each integer
        while (ss >> val) {
            // to get value between 0 and 1
            oneline.push_back(val / 255.0);
            //oneline.push_back(val);
            // If the next token is a comma, ignore it and move on
            if (ss.peek() == ',')
                ss.ignore();
        }
        result.push_back(oneline);
    }

    // Close file
    myFile.close();
    return result;
}

template <typename t>
void print_array(const t *array, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        std::cout << array[i] << ", ";
    }
    std::cout << std::endl;
}

template <typename t>
void print_vec(const std::vector<t> &vec)
{
    for (auto a : vec) {
        std::cout << a;
    }
    std::cout << std::endl;
}

std::vector<int> get_indices(size_t input_items)
{
    std::vector<int> vec(input_items);

    // initialize vec
    for (uint i = 0; i < input_items; ++i) {
        vec[i] = i;
    }
    return vec;
}

template <typename T>
int argmax(const std::vector<T> &vec)
{
    return std::distance(vec.begin(), max_element(vec.begin(), vec.end()));
}

bool is_correct(const std::vector<double> &res, const std::vector<int> &desired_res)
{
    auto num_res = argmax(res);
    auto num_des_res = argmax(desired_res);
    return num_res == num_des_res;
}

void run_and_output(Network &netw, const vec2d &input, std::string filename)
{
    // creates file with the results
    std::ofstream f(filename);
    int result;
    for (const std::vector<double> &vec : input) {
        netw.set_input(vec);
        netw.forward_pass();
        result = argmax(netw.get_output().back());
        f << result << "\n";
    }
    f.close();
}

void parallel_batch(const uint batch_counter, const uint offset, Network netw, const indexvec &train_idx, const vec2d &input, const ivec2d &desired_result, const uint iterations, std::atomic<double> &err_sum)
{
    for (uint i = 0; i < iterations; ++i) {
        auto index = train_idx[batch_counter + offset + i];
        netw.set_desired_res(desired_result[index]);
        netw.set_input(input[index]);
        netw.compute_network();
        err_sum = err_sum + netw.get_cross_entropy_error();
    }
}

void learn(Network &netw, const vec2d &input, const ivec2d &desired_result)
{
    indexvec train_idx(input.size(), 0);
    std::iota(train_idx.begin() + 1, train_idx.end(), 1);

    double DLR = LEARNING_RATE;
    double DM = MOMENTUM;

    std::atomic<double> error_sum(0);
    std::atomic<double> epoch_correct_predictions(0);
    double last_epoch_error_avg = 9999.0;

    for (uint t = 0; t < EPOCHS; t++) {
        // shuffle the indices for every epoch
        srand(0);
        std::random_shuffle(train_idx.begin(), train_idx.end());

        uint batch_counter = 0;
        while (batch_counter + BATCH_SIZE < train_idx.size()) {
#pragma omp parallel for
            for (uint i = 0; i < THREADS; ++i) {
                parallel_batch(batch_counter, BATCH_SIZE / THREADS * i, netw, train_idx, input, desired_result, BATCH_SIZE / THREADS, error_sum);
            }
            netw.update_weights_and_reset();
            batch_counter += BATCH_SIZE;
        }

        double error_avg = error_sum / train_idx.size();

        {
            double change = last_epoch_error_avg - error_avg;
            // End when good results
            if (change < DLR / 4) {
                DM += (0.96 - DM) / 2;
                DLR /= 2;
            }
        }
        last_epoch_error_avg = error_avg;

        // Reset
        error_sum = 0.0;
        epoch_correct_predictions = 0.0;
    }

    run_and_output(netw, input, "train_predictions.csv");
}

double get_mean(const std::vector<double> &v)
{
    double s = 0;
    for (auto i : v) {
        s += i;
    }
    return s / v.size();
}

double get_deviation(const std::vector<double> &v, double mean)
{
    double s = 0;
    for (auto i : v) {
        s += (i - mean) * (i - mean);
    }
    return sqrt(s / (v.size()));
}

void preprocess(vec2d &input)
{
    for (std::vector<double> &vec : input) {
        double mean = get_mean(vec);
        double deviation = get_deviation(vec, mean);
        for (double &item : vec) {
            item = (item - mean) / deviation;
        }
    }
}

int main()
{
    vec3d weights;
    atomic3d delta_weight;
    vec3d delta_acc;

    auto netw = Network(weights, delta_weight, delta_acc);
    auto train_input = read_csv_input(CSV_TRAIN_INPUT);

    // train outputs {{0,0, 1,0,0...}} - the index of one coresponds to the output number
    auto train_output = read_csv_output(CSV_TRAIN_RESULT);
    preprocess(train_input);
    learn(netw, train_input, train_output);

    // Test set run
    auto test_input = read_csv_input(CSV_TEST_INPUT);
    preprocess(test_input);
    run_and_output(netw, test_input, "test_predictions.csv");
    return 0;
}
