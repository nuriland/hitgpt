#include <err.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "bigram.h"
#include "tok.h"

typedef struct {
	Fold fold;
	const char *path;
	int echo;
} Args;

static _Noreturn void usage(void) {
	fputs("usage: gpt corpus [-e] [-m min] [-f prefix,...] file\n"
	      "       gpt bigram [-m min] [-f prefix,...] file\n",
	      stderr);
	exit(2);
}

static void addprefixes(Fold *fold, char *list) {
	for (char *s = strtok(list, ","); s != NULL; s = strtok(NULL, ",")) {
		if (fold->n == MAXFOLD) errx(1, "-f lists more than %d prefixes", MAXFOLD);
		fold->prefix[fold->n++] = s;
	}
}

// parse reads the flags in opts, and the one file every command takes.
static Args parse(int argc, char **argv, const char *opts) {
	Args a = {.fold = {.min = 1}};
	int ch;
	while ((ch = getopt(argc, argv, opts)) != -1) {
		switch (ch) {
		case 'e':
			a.echo = 1;
			break;
		case 'm':
			a.fold.min = atoi(optarg);
			break;
		case 'f':
			addprefixes(&a.fold, optarg);
			break;
		default:
			usage();
		}
	}
	if (argc - optind != 1) usage();
	a.path = argv[optind];
	return a;
}

static void echo(const Corpus *c) {
	for (int s = 0; s < c->nseq; s++) {
		for (int i = c->start[s]; i < c->start[s + 1]; i++) {
			if (i > c->start[s]) putchar(' ');
			fputs(c->vocab[c->ids[i]], stdout);
		}
		putchar('\n');
	}
}

static void printvocab(const Corpus *c) {
	for (int i = 0; i < c->nvocab; i++) printf("%5d %8d  %s\n", i, c->count[i], c->vocab[i]);
}

static int corpus(int argc, char **argv) {
	Args a = parse(argc, argv, "em:f:");
	Corpus c = corpus_load(a.path, &a.fold);
	corpus_report(&c, &a.fold, stderr);
	if (a.echo)
		echo(&c);
	else
		printvocab(&c);
	return 0;
}

// bigram prints the val loss of guessing uniformly, by how common a word is, and by the word before, at a few k
static int bigram(int argc, char **argv) {
	Args a = parse(argc, argv, "m:f:");
	Corpus c = corpus_load(a.path, &a.fold);
	corpus_report(&c, &a.fold, stderr);
	if (c.nseq < VALEVERY) errx(1, "%s: %d sequences, and the val split needs %d", a.path, c.nseq, VALEVERY);

	Bigram m = bigram_count(&c);
	static const double ks[] = {1, 0.1, 0.01, 0.001};
	double bar = INFINITY, bestk = 0;

	printf("uniform  %.3f\n\n", log(c.nvocab));
	printf("k        unigram  bigram\n");
	for (size_t i = 0; i < sizeof ks / sizeof *ks; i++) {
		Loss loss = bigram_loss(&m, &c, ks[i]);
		printf("%-8g %7.3f %7.3f\n", ks[i], loss.unigram, loss.bigram);
		if (loss.bigram < bar) {
			bar = loss.bigram;
			bestk = ks[i];
		}
	}
	printf("\nthe bar: %.3f, perplexity %.2f, at k = %g\n", bar, exp(bar), bestk);
	return 0;
}

int main(int argc, char **argv) {
	if (argc < 2) usage();
	if (strcmp(argv[1], "corpus") == 0) return corpus(argc - 1, argv + 1);
	if (strcmp(argv[1], "bigram") == 0) return bigram(argc - 1, argv + 1);
	usage();
}
