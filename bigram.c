#include "bigram.h"

#include <err.h>
#include <math.h>
#include <stdlib.h>

Bigram bigram_count(const Corpus *c) {
	int n = c->nvocab;
	Bigram m = {.n = n};
	m.pair = calloc((size_t)n * n, sizeof *m.pair);
	if (m.pair == NULL) err(1, "calloc");

	m.from = calloc(n, sizeof *m.from);
	if (m.from == NULL) err(1, "calloc");

	m.to = calloc(n, sizeof *m.to);
	if (m.to == NULL) err(1, "calloc");

	for (int s = 0; s < c->nseq; s++) {
		if (corpus_isval(s)) continue;
		for (int i = c->start[s] + 1; i < c->start[s + 1]; i++) {
			int a = c->ids[i - 1];
			int b = c->ids[i];
			m.pair[(size_t)a * n + b]++;
			m.from[a]++;
			m.to[b]++;
			m.total++;
		}
	}
	return m;
}

Loss bigram_loss(const Bigram *m, const Corpus *c, double k) {
	Loss loss = {0};
	int n = 0;
	for (int s = 0; s < c->nseq; s++) {
		if (!corpus_isval(s)) continue;
		for (int i = c->start[s] + 1; i < c->start[s + 1]; i++) {
			int a = c->ids[i - 1];
			int b = c->ids[i];
			loss.unigram -= log((m->to[b] + k) / (m->total + k * m->n));
			loss.bigram -= log((m->pair[(size_t)a * m->n + b] + k) / (m->from[a] + k * m->n));
			n++;
		}
	}
	loss.unigram /= n;
	loss.bigram /= n;
	return loss;
}
