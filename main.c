#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tok.h"

static _Noreturn void usage(void) {
	fputs("usage: gpt corpus [-e] [-m min] [-f prefix,...] file\n", stderr);
	exit(2);
}

static void addprefixes(Fold *fold, char *list) {
	for (char *s = strtok(list, ","); s != NULL; s = strtok(NULL, ",")) {
		if (fold->n == MAXFOLD) errx(1, "-f lists more than %d prefixes", MAXFOLD);
		fold->prefix[fold->n++] = s;
	}
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
	Fold fold = {.min = 1};
	int echoing = 0;
	int ch;
	while ((ch = getopt(argc, argv, "em:f:")) != -1) {
		switch (ch) {
		case 'e':
			echoing = 1;
			break;
		case 'm':
			fold.min = atoi(optarg);
			break;
		case 'f':
			addprefixes(&fold, optarg);
			break;
		default:
			usage();
		}
	}
	argc -= optind;
	argv += optind;
	if (argc != 1) usage();

	Corpus c = corpus_load(argv[0], &fold);
	corpus_report(&c, &fold, stderr);
	if (echoing)
		echo(&c);
	else
		printvocab(&c);
	return 0;
}

int main(int argc, char **argv) {
	if (argc < 2 || strcmp(argv[1], "corpus") != 0) usage();
	return corpus(argc - 1, argv + 1);
}
