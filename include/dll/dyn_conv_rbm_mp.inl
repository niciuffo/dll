//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cstddef>
#include <ctime>
#include <random>

#include "cpp_utils/assert.hpp"     //Assertions
#include "cpp_utils/stop_watch.hpp" //Performance counter
#include "cpp_utils/maybe_parallel.hpp"
#include "cpp_utils/static_if.hpp"

#include "etl/etl.hpp"

#include "standard_conv_rbm.hpp" //The base class
#include "util/timers.hpp"       //auto_timer
#include "util/checks.hpp"       //nan_check
#include "rbm_tmp.hpp"           // static_if macros

namespace dll {

/*!
 * \brief Convolutional Restricted Boltzmann Machine
 *
 * This follows the definition of a CRBM by Honglak Lee.
 */
template <typename Desc>
struct dyn_conv_rbm_mp final : public standard_conv_rbm<dyn_conv_rbm_mp<Desc>, Desc> {
    using desc      = Desc;
    using weight    = typename desc::weight;
    using this_type = dyn_conv_rbm_mp<desc>;
    using base_type = standard_conv_rbm<this_type, desc>;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit  = desc::hidden_unit;
    static constexpr const unit_type pooling_unit = desc::pooling_unit;

    static constexpr const bool dbn_only = layer_traits<this_type>::is_dbn_only();

    //TODO CHECK
    template <std::size_t B>
    using input_batch_t = etl::fast_dyn_matrix<weight, B, 1>;

    using w_type = etl::dyn_matrix<weight, 4>;
    using b_type = etl::dyn_vector<weight>;
    using c_type = etl::dyn_vector<weight>;

    using input_t      = typename rbm_base_traits<this_type>::input_t;
    using output_t     = typename rbm_base_traits<this_type>::output_t;
    using input_one_t  = typename rbm_base_traits<this_type>::input_one_t;
    using output_one_t = typename rbm_base_traits<this_type>::output_one_t;

    w_type w; //!< shared weights
    b_type b; //!< hidden biases bk
    c_type c; //!< visible single bias c

    std::unique_ptr<w_type> bak_w; //!< backup shared weights
    std::unique_ptr<b_type> bak_b; //!< backup hidden biases bk
    std::unique_ptr<c_type> bak_c; //!< backup visible single bias c

    etl::dyn_matrix<weight, 3> v1; //visible units

    etl::dyn_matrix<weight, 3> h1_a; ///< Activation probabilities of reconstructed hidden units
    etl::dyn_matrix<weight, 3> h1_s; ///< Sampled values of reconstructed hidden units

    etl::dyn_matrix<weight, 3> p1_a; ///< Pooled Activation probabilities of reconstructed hidden units
    etl::dyn_matrix<weight, 3> p1_s; ///< Pooled Sampled values of reconstructed hidden units

    etl::dyn_matrix<weight, 3> v2_a; ///< Activation probabilities of reconstructed visible units
    etl::dyn_matrix<weight, 3> v2_s; ///< Sampled values of reconstructed visible units

    etl::dyn_matrix<weight, 3> h2_a; ///< Activation probabilities of reconstructed hidden units
    etl::dyn_matrix<weight, 3> h2_s; ///< Sampled values of reconstructed hidden units

    etl::dyn_matrix<weight, 3> p2_a; ///< Pooled Activation probabilities of reconstructed hidden units
    etl::dyn_matrix<weight, 3> p2_s; ///< Pooled Sampled values of reconstructed hidden units

    //Convolution data

    static constexpr const std::size_t V_CV_CHANNELS = 2;
    static constexpr const std::size_t H_CV_CHANNELS = 2;

    etl::dyn_matrix<weight, 4> v_cv; ///< Temporary convolution
    etl::dyn_matrix<weight, 3> h_cv; ///< Temporary convolution

    size_t nv1; ///< The first visible dimension
    size_t nv2; ///< The second visible dimension
    size_t nh1; ///< The first output dimension
    size_t nh2; ///< The second output dimension
    size_t nc;  ///< The number of input channels
    size_t k;   ///< The number of filters
    size_t p_c;

    size_t nw1; ///< The first dimension of the filters
    size_t nw2; ///< The second dimension of the filters
    size_t np1;
    size_t np2;

    size_t batch_size = 25;

    mutable cpp::thread_pool<!layer_traits<this_type>::is_serial()> pool;

    dyn_conv_rbm_mp() : base_type(), pool(etl::threads) {
        // Nothing else to init
    }

    void prepare_input(input_one_t& input) const {
        input = input_one_t(nc, nv1, nv2);
    }

    void init_layer(size_t nc, size_t nv1, size_t nv2, size_t k, size_t nh1, size_t nh2, size_t p_c){
        this->nv1 = nv1;
        this->nv2 = nv2;
        this->nh1 = nh1;
        this->nh2 = nh2;
        this->nc = nc;
        this->k = k;
        this->p_c = p_c;

        this->nw1 = nv1 - nh1 + 1;
        this->nw2 = nv2 - nh2 + 1;

        this->np1 = nh1 / p_c;
        this->np2 = nh2 / p_c;

#ifdef ETL_CUDNN_MODE
        w = etl::dyn_matrix<weight, 4>(k, nc, nw1, nw2);
#else
        w = etl::dyn_matrix<weight, 4>(nc, k, nw1, nw2);
#endif

        b = etl::dyn_vector<weight>(k);
        c = etl::dyn_vector<weight>(nc);

        v1 = etl::dyn_matrix<weight, 3>(nc, nv1, nv2);

        h1_a = etl::dyn_matrix<weight, 3>(k, nh1, nh2);
        h1_s = etl::dyn_matrix<weight, 3>(k, nh1, nh2);

        p1_a = etl::dyn_matrix<weight, 3>(k, np1, np2);
        p1_s = etl::dyn_matrix<weight, 3>(k, np1, np2);

        v2_a = etl::dyn_matrix<weight, 3>(nc, nv1, nv2);
        v2_s = etl::dyn_matrix<weight, 3>(nc, nv1, nv2);

        h2_a = etl::dyn_matrix<weight, 3>(k, nh1, nh2);
        h2_s = etl::dyn_matrix<weight, 3>(k, nh1, nh2);

        p2_a = etl::dyn_matrix<weight, 3>(k, np1, np2);
        p2_s = etl::dyn_matrix<weight, 3>(k, np1, np2);

        v_cv = etl::dyn_matrix<weight, 4>(V_CV_CHANNELS, k, nh1, nh2);
        h_cv = etl::dyn_matrix<weight, 3>(H_CV_CHANNELS, nv1, nv2);

        if (is_relu(hidden_unit)) {
            w = etl::normal_generator(0.0, 0.01);
            b = 0.0;
            c = 0.0;
        } else {
            w = 0.01 * etl::normal_generator();
            b = -0.1;
            c = 0.0;
        }
    }

    std::size_t input_size() const noexcept {
        return nv1 * nv2 * nc;
    }

    std::size_t output_size() const noexcept {
        return np1 * np2 * k;
    }

    std::size_t parameters() const noexcept {
        return nc * k * nw1 * nw2;
    }

    std::string to_short_string() const {
        char buffer[1024];
        snprintf(
            buffer, 1024, "CRBM_MP(dyn)(%s): %lux%lux%lu -> (%lux%lu) -> %lux%lux%lu -> %lux%lux%lu",
            to_string(hidden_unit).c_str(), nv1, nv2, nc, nw1, nw2, nh1, nh2, k, np1, np2, k);
        return {buffer};
    }

    void display() const {
        std::cout << to_short_string() << std::endl;
    }

    void backup_weights() {
        unique_safe_get(bak_w) = w;
        unique_safe_get(bak_b) = b;
        unique_safe_get(bak_c) = c;
    }

    void restore_weights() {
        w = *bak_w;
        b = *bak_b;
        c = *bak_c;
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2& v_s) const {
        etl::dyn_matrix<weight, 4> v_cv(V_CV_CHANNELS, k, nh1, nh2); //Temporary convolution
        activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, v_cv);
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2>
    void activate_visible(const H1& h_a, const H2& h_s, V1&& v_a, V2&& v_s) const {
        etl::dyn_matrix<weight, 3> h_cv(H_CV_CHANNELS, nv2, nv2); //Temporary convolution
        activate_visible<P, S>(h_a, h_s, std::forward<V1>(v_a), std::forward<V2>(v_s), h_cv);
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2, typename VCV>
    void activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2& /*v_s*/, VCV&& v_cv) const {
        dll::auto_timer timer("dyn_crbm:activate_hidden");

        static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit), "Invalid hidden unit type");
        static_assert(P, "Computing S without P is not implemented");

        using namespace etl;

        auto b_rep = etl::force_temporary(etl::rep(b, nh1, nh2));

        base_type::template compute_vcv(*this, v_a, v_cv, w);

        H_PROBS2(unit_type::BINARY, unit_type::BINARY, f(h_a) = etl::p_max_pool_h(b_rep + v_cv(1), p_c, p_c));
        H_PROBS2(unit_type::BINARY, unit_type::GAUSSIAN, f(h_a) = etl::p_max_pool_h((1.0 / (0.1 * 0.1)) >> (b_rep + v_cv(1)), p_c, p_c));
        H_PROBS(unit_type::RELU, f(h_a) = f(h_a) = max(b_rep + v_cv(1), 0.0));
        H_PROBS(unit_type::RELU6, f(h_a) = f(h_a) = min(max(b_rep + v_cv(1), 0.0), 6.0));
        H_PROBS(unit_type::RELU1, f(h_a) = f(h_a) = min(max(b_rep + v_cv(1), 0.0), 1.0));

        H_SAMPLE_PROBS(unit_type::BINARY, f(h_s) = bernoulli(h_a));
        H_SAMPLE_PROBS(unit_type::RELU, f(h_s) = max(logistic_noise(b_rep + v_cv(1)), 0.0));
        H_SAMPLE_PROBS(unit_type::RELU6, f(h_s) = ranged_noise(h_a, 6.0));
        H_SAMPLE_PROBS(unit_type::RELU1, f(h_s) = ranged_noise(h_a, 1.0));

        nan_check_deep(h_a);

        if (S) {
            nan_check_deep(h_s);
        }
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2, typename HCV>
    void activate_visible(const H1& /*h_a*/, const H2& h_s, V1&& v_a, V2&& v_s, HCV&& h_cv) const {
        dll::auto_timer timer("dyn_crbm:activate_visible");

        static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN, "Invalid visible unit type");
        static_assert(P, "Computing S without P is not implemented");

        using namespace etl;

        base_type::template compute_hcv(*this, h_s, h_cv, w, [&](std::size_t channel) {
            V_PROBS(unit_type::BINARY, f(v_a)(channel) = sigmoid(c(channel) + h_cv(1)));
            V_PROBS(unit_type::GAUSSIAN, f(v_a)(channel) = c(channel) + h_cv(1));
        });

        V_SAMPLE_PROBS(unit_type::BINARY, f(v_s) = bernoulli(v_a));
        V_SAMPLE_PROBS(unit_type::GAUSSIAN, f(v_s) = normal_noise(v_a));

        nan_check_etl(v_a);

        if (S) {
            nan_check_deep(v_s);
        }
    }

    template <bool P = true, bool S = true, typename Po, typename V>
    void activate_pooling(Po& p_a, Po& p_s, const V& v_a, const V&) const {
        dll::auto_timer timer("crbm:mp:activate_pooling");

        static_assert(pooling_unit == unit_type::BINARY, "Invalid pooling unit type");
        static_assert(P, "Computing S without P is not implemented");

        etl::dyn_matrix<weight, 4> v_cv(2, k, nh1, nh2); //Temporary convolution

        base_type::template compute_vcv(*this, v_a, v_cv, w);

        if (pooling_unit == unit_type::BINARY) {
            p_a = etl::p_max_pool_p(etl::rep(b, nh1, nh2) + v_cv(1), p_c, p_c);
        }

        nan_check_etl(p_a);

        if (S) {
            if (pooling_unit == unit_type::BINARY) {
                p_s = r_bernoulli(p_a);
            }

            nan_check_etl(p_s);
        }
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2, typename VCV>
    void batch_activate_hidden(H1&& h_a, H2&& h_s, const V1& v_a, const V2& /*v_s*/, VCV&& v_cv) const {
        dll::auto_timer timer("dyn_crbm:batch_activate_hidden");

        static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit), "Invalid hidden unit type");
        static_assert(P, "Computing S without P is not implemented");

        using namespace etl;

        const auto Batch = etl::dim<0>(h_a);

        cpp_assert(etl::dim<0>(h_s) == Batch, "The number of batch must be consistent");
        cpp_assert(etl::dim<0>(v_a) == Batch, "The number of batch must be consistent");
        cpp_assert(etl::dim<0>(v_cv) == Batch, "The number of batch must be consistent");
        cpp_unused(Batch);

        base_type::template batch_compute_vcv(*this, pool, v_a, v_cv, w, [&](std::size_t batch) {
            H_PROBS2(unit_type::BINARY, unit_type::BINARY, f(h_a)(batch) = etl::p_max_pool_h(etl::rep(b, nh1, nh2) + v_cv(batch)(1), p_c, p_c));
            H_PROBS2(unit_type::BINARY, unit_type::GAUSSIAN, f(h_a)(batch) = etl::p_max_pool_h((1.0 / (0.1 * 0.1)) >> (etl::rep(b, nh1, nh2) + v_cv(batch)(1)), p_c, p_c));
            H_PROBS(unit_type::RELU, f(h_a)(batch) = max(etl::rep(b, nh1, nh2) + v_cv(batch)(1), 0.0));
            H_PROBS(unit_type::RELU6, f(h_a)(batch) = min(max(etl::rep(b, nh1, nh2) + v_cv(batch)(1), 0.0), 6.0));
            H_PROBS(unit_type::RELU1, f(h_a)(batch) = min(max(etl::rep(b, nh1, nh2) + v_cv(batch)(1), 0.0), 1.0));

            H_SAMPLE_PROBS(unit_type::RELU, f(h_s)(batch) = max(logistic_noise(etl::rep(b, nh1, nh2) + v_cv(batch)(1)), 0.0));
        });

        H_SAMPLE_PROBS(unit_type::BINARY, f(h_s) = bernoulli(h_a));
        H_SAMPLE_PROBS(unit_type::RELU6, f(h_s) = ranged_noise(h_a, 6.0));
        H_SAMPLE_PROBS(unit_type::RELU1, f(h_s) = ranged_noise(h_a, 1.0));

        nan_check_deep(h_a);

        if (S) {
            nan_check_deep(h_s);
        }
    }

    template <bool P = true, bool S = true, typename H1, typename H2, typename V1, typename V2, typename HCV>
    void batch_activate_visible(const H1& /*h_a*/, const H2& h_s, V1&& v_a, V2&& v_s, HCV&& h_cv) const {
        dll::auto_timer timer("dyn_crbm:batch_activate_visible");

        static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN, "Invalid visible unit type");
        static_assert(P, "Computing S without P is not implemented");

        const auto Batch = batch_size;

        cpp_assert(etl::dim<0>(h_s) == Batch, "The number of batch must be consistent");
        cpp_assert(etl::dim<0>(v_a) == Batch, "The number of batch must be consistent");
        cpp_assert(etl::dim<0>(v_s) == Batch, "The number of batch must be consistent");
        cpp_assert(etl::dim<0>(h_cv) == Batch, "The number of batch must be consistent");
        cpp_unused(Batch);

        base_type::template batch_compute_hcv(*this, pool, h_s, h_cv, w, [&](std::size_t batch, std::size_t channel) {
            V_PROBS(unit_type::BINARY, f(v_a)(batch)(channel) = etl::sigmoid(c(channel) + h_cv(batch)(1)));
            V_PROBS(unit_type::GAUSSIAN, f(v_a)(batch)(channel) = c(channel) + h_cv(batch)(1));
        });

        V_SAMPLE_PROBS(unit_type::BINARY, f(v_s) = bernoulli(v_a));
        V_SAMPLE_PROBS(unit_type::GAUSSIAN, f(v_s) = normal_noise(v_a));

        nan_check_deep(v_a);

        if (S) {
            nan_check_deep(v_s);
        }
    }

    template <typename V, typename H, cpp_enable_if(etl::is_etl_expr<V>::value)>
    weight energy(const V& v, const H& h) const {
        etl::dyn_matrix<weight, 4> v_cv(V_CV_CHANNELS, k, nh1, nh2); //Temporary convolution

        if (desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY) {
            //Definition according to Honglak Lee
            //E(v,h) = - sum_k (hk (Wk*v) + bk hk) - c sum_v v

            base_type::template compute_vcv(*this, v, v_cv, w);

            return -etl::sum(c >> etl::sum_r(v)) - etl::sum((h >> v_cv(1)) + (etl::rep(b, nh1, nh2) >> h));
        } else if (desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY) {
            //Definition according to Honglak Lee / Mixed with Gaussian
            //E(v,h) = - sum_k (hk (Wk*v) + bk hk) - sum_v ((v - c) ^ 2 / 2)

            base_type::template compute_vcv(*this, v, v_cv, w);

            return -sum(etl::pow(v - etl::rep(c, nv1, nv2), 2) / 2.0) - etl::sum((h >> v_cv(1)) + (etl::rep(b, nh1, nh2) >> h));
        } else {
            return 0.0;
        }
    }

    template <typename V, typename H, cpp_disable_if(etl::is_etl_expr<V>::value)>
    weight energy(const V& v, const H& h) const {
        etl::dyn_matrix<weight, 3> ev(nc, nv1, nv2);
        etl::dyn_matrix<weight, 3> eh(k, nh1, nh2);

        ev = v;
        eh = h;

        return energy(ev, eh);
    }

    template <typename V>
    weight free_energy_impl(const V& v) const {
        etl::dyn_matrix<weight, 4> v_cv(V_CV_CHANNELS, k, nh1, nh2); //Temporary convolution

        if (desc::visible_unit == unit_type::BINARY && desc::hidden_unit == unit_type::BINARY) {
            //Definition computed from E(v,h)

            base_type::template compute_vcv(*this, v, v_cv, w);

            auto x = etl::rep(b, nh1, nh2) + v_cv(1);

            return -etl::sum(c >> etl::sum_r(v)) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else if (desc::visible_unit == unit_type::GAUSSIAN && desc::hidden_unit == unit_type::BINARY) {
            //Definition computed from E(v,h)

            base_type::template compute_vcv(*this, v, v_cv, w);

            auto x = etl::rep(b, nh1, nh2) + v_cv(1);

            return -sum(etl::pow(v - etl::rep(c, nv1, nv2), 2) / 2.0) - etl::sum(etl::log(1.0 + etl::exp(x)));
        } else {
            return 0.0;
        }
    }

    template <typename V>
    weight free_energy(const V& v) const {
        etl::dyn_matrix<weight, 3> ev(nc, nv1, nv2);
        ev = v;
        return free_energy_impl(ev);
    }

    weight free_energy() const {
        return free_energy_impl(v1);
    }

    //Utilities for DBNs

    template <typename Input>
    output_t prepare_output(std::size_t samples) const {
        output_t output;
        output.reserve(samples);
        for(size_t i = 0; i < samples; ++i){
            output.emplace_back(k, np1, np2);
        }
        return output;
    }

    template <typename Input>
    output_one_t prepare_one_output() const {
        return output_one_t(k, nh1, nh2);
    }

    void activate_hidden(output_one_t& h_a, const input_one_t& input) const {
        activate_pooling<true, false>(h_a, h_a, input, input);
    }

    void activate_many(const input_t& input, output_t& h_a, output_t& h_s) const {
        for (std::size_t i = 0; i < input.size(); ++i) {
            activate_one(input[i], h_a[i], h_s[i]);
        }
    }

    void activate_many(const input_t& input, output_t& h_a) const {
        for (std::size_t i = 0; i < input.size(); ++i) {
            activate_one(input[i], h_a[i]);
        }
    }
};

/*!
 * \brief Simple traits to pass information around from the real
 * class to the CRTP class.
 */
template <typename Desc>
struct rbm_base_traits<dyn_conv_rbm_mp<Desc>> {
    using desc      = Desc;
    using weight    = typename desc::weight;

    using input_one_t   = etl::dyn_matrix<weight, 3>;
    using output_one_t  = etl::dyn_matrix<weight, 3>;
    using input_t       = std::vector<input_one_t>;
    using output_t      = std::vector<output_one_t>;
};

} //end of dll namespace
