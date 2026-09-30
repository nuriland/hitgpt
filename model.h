#pragma once

#include <stdint.h>

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

// model_embed turns n ids, n <= T, into n vectors of C numbers each: x[t] = wte[id] + wpe[t].
void model_embed(const Model *m, const uint16_t *ids, int n, float *x);

// layernorm normalizes each of n vectors of C numbers on its own, to mean 0 and standard deviation 1, 
// then scales it by w and shifts it by b.
void layernorm(float *out, const float *x, const float *w, const float *b, int n, int C);
