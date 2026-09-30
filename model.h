#pragma once

#include <stdint.h>

#include "tok.h"

typedef struct {
	int V; // vocab size
	int C; // width, the numbers per vector
	int T; // context, the most words it sees at once
} Config;

// Model is every weight in one array, with a pointer to where each table starts.
typedef struct {
	Config cfg;
	float *params;
	int nparams;
	float *wte;  // V × C: a row per word
	float *wpe;  // T × C: a row per position
	float *lnfw; // C: the final LayerNorm's scale Y
	float *lnfb; // C: and its shift B
} Model;

// model_init draws the tables from a normal distribution with standard deviation 0.02, and starts
// LayerNorm at Y = 1 and B = 0. The same seed gives the same weights.
Model model_init(Config cfg, uint64_t seed);

// model_forward runs n ids, n <= T, through the model, giving n rows of V logits.
// At each position, it gives a score for every word being the next one.
void model_forward(const Model *m, const uint16_t *ids, int n, float *logits);

// logprob is ln p[target] under the softmax of V logits.
double logprob(const float *logits, int V, int target);

// model_loss is the mean -ln p[the real next word] over the val sequences, each fed on its own from its first word
double model_loss(const Model *m, const Corpus *c);
