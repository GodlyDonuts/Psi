// main.cpp — psi-nano training driver (CPU / Mac prototype).
//
// Char-level GPT trained on a small embedded corpus. Demonstrates the full custom stack
// end to end: tokenize -> embed -> transformer -> cross-entropy -> autograd backward ->
// AdamW -> sample. Slow on purpose (naive double-precision CPU ops); kernels come later.
//
// Usage:  psi_nano [steps]   (default 2000)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "model.hpp"

using namespace psi;

// Small self-contained corpus (lowercase to keep the vocab tiny so a CPU model can learn fast).
static const std::string CORPUS =
    "psi is a small language model. it learns to predict the next character in a "
    "sequence. the model is built from scratch with a custom autograd engine. every "
    "operation knows how to compute its own gradient. we train the model on a tiny "
    "corpus of text. the loss goes down as the model learns. when the loss is low, the "
    "model can generate text that looks like the data. this is only a prototype. later "
    "we will make it fast with custom kernels and run it on a powerful gpu. for now it "
    "runs slowly on a laptop. the goal is to understand every part of the system. small "
    "models can still be smart. we measure quality per bit. the smallest model that is "
    "still clever wins. one character at a time, the model learns the shape of language. ";

int main(int argc, char** argv) {
    // args: a number sets training steps; "chat" enters an interactive prompt after training.
    bool chat = false;
    int steps = 2000;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "chat" || a == "--chat") chat = true;
        else steps = std::atoi(argv[i]);
    }

    // ---- tokenizer: char-level ----
    std::vector<char> id2ch;
    std::unordered_map<char, int> ch2id;
    for (char c : CORPUS)
        if (!ch2id.count(c)) { ch2id[c] = (int)id2ch.size(); id2ch.push_back(c); }
    int V = (int)id2ch.size();

    std::vector<int> data;
    data.reserve(CORPUS.size());
    for (char c : CORPUS) data.push_back(ch2id[c]);

    Config cfg{V, /*d_model*/ 64, /*n_layers*/ 2, /*block*/ 32, /*hidden*/ 256};
    std::mt19937 rng(1234);
    GPT model(cfg, rng);
    AdamW opt(model.params());

    const int  batch = 8;
    const real lr    = 1e-3;

    int nparams = 0;
    for (auto& p : model.params()) nparams += p.numel();
    std::printf("psi-nano: vocab=%d  d=%d  layers=%d  block=%d  params=%d  corpus=%zu chars\n",
                V, cfg.d_model, cfg.n_layers, cfg.block, nparams, CORPUS.size());
    std::printf("training: %d steps, batch %d, lr %.0e\n\n", steps, batch, lr);

    std::uniform_int_distribution<int> startpick(0, (int)data.size() - cfg.block - 2);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step <= steps; ++step) {
        // Mini-batch: sum cross-entropy over `batch` random windows, then average.
        std::vector<Tensor> losses;
        for (int b = 0; b < batch; ++b) {
            int i = startpick(rng);
            std::vector<int> ids(data.begin() + i, data.begin() + i + cfg.block);
            std::vector<int> tgt(data.begin() + i + 1, data.begin() + i + 1 + cfg.block);
            losses.push_back(cross_entropy(model.forward(ids), tgt));
        }
        Tensor loss = losses[0];
        for (size_t k = 1; k < losses.size(); ++k) loss = add(loss, losses[k]);
        loss = scalar_mul(loss, 1.0 / batch);

        opt.zero_grad();
        loss.backward();
        opt.step(lr);

        if (step % 50 == 0) {
            double el = std::chrono::duration<double>(
                            std::chrono::high_resolution_clock::now() - t0).count();
            std::printf("step %5d   loss %.4f   (%.1fs)\n", step, loss.data()[0], el);
        }
        if (step > 0 && step % 500 == 0) {
            std::string s = generate(model, {ch2id['t'], ch2id['h'], ch2id['e']},
                                     120, 0.8, rng, id2ch);
            std::printf("  sample: \"the%s\"\n", s.c_str());
        }
    }

    std::printf("\nfinal samples (seed \"the \"):\n");
    std::vector<int> seed = {ch2id['t'], ch2id['h'], ch2id['e'], ch2id[' ']};
    for (int k = 0; k < 3; ++k) {
        std::string s = generate(model, seed, 160, 0.7, rng, id2ch);
        std::printf("  \"the %s\"\n", s.c_str());
    }

    if (chat) {
        std::printf("\n[interactive] type a lowercase seed (a-z, space, '.'); 'quit' to exit.\n> ");
        std::fflush(stdout);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "quit" || line == "exit") break;
            std::vector<int> ctx;                       // keep only in-vocab chars
            for (char c : line) if (ch2id.count(c)) ctx.push_back(ch2id[c]);
            if (ctx.empty()) ctx.push_back(ch2id[' ']);
            std::string s = generate(model, ctx, 200, 0.7, rng, id2ch);
            std::printf("%s%s\n> ", line.c_str(), s.c_str());
            std::fflush(stdout);
        }
    }
    return 0;
}
