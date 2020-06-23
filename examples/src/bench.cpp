//=======================================================================
// Copyright (c) 2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define ETL_COUNTERS
#define ETL_GPU_TOOLS
#define ETL_GPU_POOL

#include <iostream>
#include <chrono>
#include <dll/neural/conv/conv_same_layer.hpp>
#include <dll/neural/dropout/dropout_layer.hpp>

#include "dll/neural/dense/dense_layer.hpp"
#include "dll/pooling/mp_layer.hpp"
#include "dll/network.hpp"
#include "dll/datasets.hpp"

double cifar10(int n) {
    // Load the dataset
    auto dataset = dll::make_cifar10_dataset(dll::batch_size<256>{}, dll::scale_pre<255>{});

    using dbn_t = dll::dbn_desc<
            dll::dbn_layers<
                    dll::conv_same_layer<3, 32, 32, 12, 5, 5, dll::relu>,
                    dll::conv_same_layer<12, 32, 32, 12, 3, 3, dll::relu>,
                    dll::mp_3d_layer<12, 32, 32, 1, 2, 2>,
                    dll::conv_same_layer<12, 16, 16, 24, 5, 5, dll::relu>,
                    dll::conv_same_layer<24, 16, 16, 24, 3, 3, dll::relu>,
                    dll::mp_3d_layer<24, 16, 16, 1, 2, 2>,
                    dll::conv_same_layer<24, 8, 8, 48, 3, 3, dll::relu>,
                    dll::conv_same_layer<48, 8, 8, 48, 3, 3, dll::relu>,
                    dll::mp_3d_layer<48, 8, 8, 1, 2, 2>,
                    dll::dense_layer<48 * 4 * 4, 64, dll::relu>,
                    dll::dense_layer<64, 10, dll::softmax>
            >,
            dll::updater<dll::updater_type::MOMENTUM>,
            dll::binary_cross_entropy,
            dll::batch_size<256>,
            dll::no_batch_display,
            dll::no_epoch_error
    >::dbn_t;

    auto dbn = std::make_unique<dbn_t>();

    dbn->learning_rate = 0.001;
    dbn->initial_momentum = 0.9;
    dbn->momentum = 0.9;
    dbn->goal = -1.0;

    auto start = std::chrono::high_resolution_clock::now();
    dbn->fine_tune(dataset.train(), n);
    auto stop = std::chrono::high_resolution_clock::now();

    dbn->evaluate(dataset.test());

    auto duration = duration_cast<std::chrono::milliseconds>(stop - start).count();

    std::cout << duration << std::endl;
    return duration;
}

double mlp(int n) {
    // Load the dataset
    auto dataset = dll::make_mnist_dataset(dll::batch_size<256>{}, dll::normalize_pre{});

    // Build the network

    using network_t = dll::network_desc<
            dll::network_layers<
                    dll::dense_layer<28 * 28, 500>,
                    dll::dropout_layer<50>,
                    dll::dense_layer<500, 1000>,
                    dll::dropout_layer<50>,
                    dll::dense_layer<1000, 1000>,
                    dll::dense_layer<1000, 10, dll::softmax>
            >
            , dll::updater<dll::updater_type::NADAM>     // Nesterov Adam (NADAM)
            , dll::binary_cross_entropy
            , dll::batch_size<256>                       // The mini-batch size
            , dll::shuffle                               // Shuffle before each epoch
            , dll::no_batch_display                      // Disable pretty print of each every batch
            , dll::no_epoch_error                        // Disable computation of the error at each epoch
    >::network_t;

    auto net = std::make_unique<network_t>();

    auto start = std::chrono::high_resolution_clock::now();
    // Train the network for performance sake
    net->train(dataset.train(), n);
    auto stop = std::chrono::high_resolution_clock::now();

    // Test the network on test set
    net->evaluate(dataset.test());

    auto duration = duration_cast<std::chrono::milliseconds>(stop - start).count();

    std::cout << duration << std::endl;
    return duration;
}

int main(int /*argc*/, char* /*argv*/ []) {
    auto const ITR = 3;
    auto const CIFAR10_EPOCHS = 10;
    auto const MLP_EPOCHS = 10;

    double cifar10_mean = 0;
    double mlp_mean = 0;

    std::cout << "cifar10" << std::endl;
    for (int i = 0; i < ITR; i++) {
        cifar10_mean += cifar10(CIFAR10_EPOCHS);
    }

    std::cout << "mlp" << std::endl;
    for (int i = 0; i < ITR; i++) {
        mlp_mean += mlp(MLP_EPOCHS);
    }

    cifar10_mean /= ITR;
    mlp_mean /= ITR;

    std::cout << "cifar10 mean after " << ITR << " iterations: " << cifar10_mean << std::endl;
    std::cout << "mlp mean after " << ITR << " iterations: " << mlp_mean << std::endl;
}