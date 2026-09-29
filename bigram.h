#pragma once

#include "tok.h"

// Bigram counts, over the train sequences, how often each word follows each other word
typedef struct {
	int n;     // the vocab size
	int *pair; // pair[a*n + b], i.e. how often b follows a
	int *from; // from[a], i.e. how often a is followed by anything
	int *to;   // to[b], i.e. how often b follows anything
	int total; // pairs counted
} Bigram;

// Loss is the mean surprise, -ln P(word), over every val word after a sequence's first. 
// The unigram guesses a word by how common it is, the bigram by the word before.
typedef struct {
	double unigram, bigram;
} Loss;

Bigram bigram_count(const Corpus *c);

Loss bigram_loss(const Bigram *m, const Corpus *c, double k);
