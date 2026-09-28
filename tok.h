#pragma once

#include <stdint.h>
#include <stdio.h>

enum {
	MAXFOLD = 16,
	VALEVERY = 10,
};

typedef struct {
	int min, n;
	char *prefix[MAXFOLD];
} Fold;

typedef struct {
	char **vocab;        // id -> word, the most common first
	int *count;          // id -> how often it appears
	int nvocab;
	uint16_t *ids;       // every word, in the text's order
	int *start;          // sequence s is ids[start[s]..start[s+1])
	int nids, nseq;
	int unfolded;        // the vocab before folding
	int folded[MAXFOLD]; // words folded under each prefix
} Corpus;

Corpus corpus_load(const char *path, const Fold *fold);

void corpus_report(const Corpus *c, const Fold *fold, FILE *f);
