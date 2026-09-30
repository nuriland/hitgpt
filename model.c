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

// embed turns n ids into n vectors of C numbers each: x[t] = wte[id] + wpe[t].
static void embed(const Model *m, const uint16_t *ids, int n, float *x) {
	int C = m->cfg.C;
	for (int t = 0; t < n; t++) {
		const float *word = m->wte + ids[t] * C;
		const float *pos = m->wpe + t * C;
		for (int i = 0; i < C; i++) x[t * C + i] = word[i] + pos[i];
	}
}

// layernorm normalizes each of n vectors of C numbers on its own, to mean 0 and standard deviiation 1,
// then scales it by w and shifts it by b.
static void layernorm(float *out, const float *x, const float *w, const float *b, int n, int C) {
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

void model_forward(const Model *m, const uint16_t *ids, int n, float *logits) {
	int C = m->cfg.C, V = m->cfg.V;

	float *x = calloc((size_t)n * C, sizeof *x);
	if (x == NULL) err(1, "calloc");

	float *y = calloc((size_t)n * C, sizeof *y);
	if (y == NULL) err(1, "calloc");

	embed(m, ids, n, x);
	layernorm(y, x, m->lnfw, m->lnfb, n, C);

	// the scores come from the word table itself, each vector against every word's row
	for (int t = 0; t < n; t++) {
		for (int v = 0; v < V; v++) {
			float dot = 0;
			for (int i = 0; i < C; i++) dot += y[t * C + i] * m->wte[v * C + i];
			logits[t * V + v] = dot;
		}
	}
	free(x);
	free(y);
}

double logprob(const float *logits, int V, int target) {
	// ln softmax at target, which is logit[target] - ln sum e^logit
	float max = logits[0];
	for (int v = 1; v < V; v++)
		if (logits[v] > max) max = logits[v];

	double sum = 0;
	for (int v = 0; v < V; v++) sum += exp(logits[v] - max);

	return logits[target] - max - log(sum);
}

double model_loss(const Model *m, const Corpus *c) {
	int V = m->cfg.V;
	double sum = 0;
	int npred = 0;

	for (int s = 0; s < c->nseq; s++) {
		if (!corpus_isval(s)) continue;

		const uint16_t *ids = c->ids + c->start[s];
		int n = c->start[s + 1] - c->start[s];
		if (n > m->cfg.T) errx(1, "val sequence %d has %d words, more than the context of %d", s, n, m->cfg.T);

		float *logits = calloc((size_t)n * V, sizeof *logits);
		if (logits == NULL) err(1, "calloc");

		model_forward(m, ids, n, logits);
		for (int t = 0; t + 1 < n; t++) {
			sum -= logprob(logits + t * V, V, ids[t + 1]);
			npred++;
		}
		free(logits);
	}
	return sum / npred;
}
