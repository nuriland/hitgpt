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
	m.nparams = cfg.V * cfg.C + cfg.T * cfg.C;
	m.params = calloc(m.nparams, sizeof *m.params);
	if (m.params == NULL) err(1, "calloc");

	m.wte = m.params;
	m.wpe = m.wte + cfg.V * cfg.C;

	for (int i = 0; i < m.nparams; i++) m.params[i] = 0.02 * normal(&seed);
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
