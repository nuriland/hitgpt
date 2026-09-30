#include "model.h"

#include <err.h>
#include <math.h>
#include <stdlib.h>

// splitmix64 gives 64 new random bits a call, small and fast, and the same seed gives the same bits.
static uint64_t splitmix64(uint64_t *s) {
	uint64_t z = (*s += 0x9e3779b97f4a7c15u);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9u;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebu;
	return z ^ (z >> 31);
}

static double uniform(uint64_t *s) { return (splitmix64(s) >> 11) * 0x1.0p-53; }

static double normal(uint64_t *s) {
	double u = uniform(s);
	double v = uniform(s);
	return sqrt(-2 * log(1 - u)) * cos(6.283185307179586 * v);
}

Model model_init(Config cfg, uint64_t seed) {
	Model m = {.cfg = cfg};
	int tables = cfg.V * cfg.C + cfg.T * cfg.C;

	m.nparams = tables + 2 * cfg.C;
	m.params = calloc(m.nparams, sizeof *m.params);
	if (m.params == NULL) err(1, "calloc");

	m.wte = m.params;
	m.wpe = m.wte + cfg.V * cfg.C;
	m.lnfw = m.wpe + cfg.T * cfg.C;
	m.lnfb = m.lnfw + cfg.C;

	for (int i = 0; i < tables; i++) m.params[i] = 0.02 * normal(&seed);
	for (int i = 0; i < cfg.C; i++) m.lnfw[i] = 1; // lnfb stays 0, from calloc
	return m;
}

void model_embed(const Model *m, const uint16_t *ids, int n, float *x) {
	int C = m->cfg.C;
	for (int t = 0; t < n; t++) {
		const float *word = m->wte + ids[t] * C;
		const float *pos = m->wpe + t * C;
		for (int i = 0; i < C; i++) x[t * C + i] = word[i] + pos[i];
	}
}

void layernorm(float *out, const float *x, const float *w, const float *b, int n, int C) {
	for (int t = 0; t < n; t++) {
		const float *v = x + t * C;
		float mean = 0;
		for (int i = 0; i < C; i++) mean += v[i];
		mean /= C;

		float var = 0;
		for (int i = 0; i < C; i++) var += (v[i] - mean) * (v[i] - mean);
		var /= C;
		float rstd = 1 / sqrtf(var + 1e-5f); // ε keeps a vector of all the same number from dividing by 0

		float *o = out + t * C;
		for (int i = 0; i < C; i++) o[i] = (v[i] - mean) * rstd * w[i] + b[i];
	}
}
