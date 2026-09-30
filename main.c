#include <err.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "bigram.h"
#include "model.h"
#include "tok.h"

typedef struct {
	Fold fold;
	const char *path;
	int echo;
	int width, context, seed;
} Args;

static _Noreturn void usage(void) {
	fputs("usage: gpt corpus [-e] [-m min] [-f prefix,...] file\n"
	      "       gpt bigram [-m min] [-f prefix,...] file\n"
	      "       gpt forward [-m min] [-f prefix,...] [-C width] [-T context] [-s seed] file\n",
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
	Args a = {.fold = {.min = 1}, .width = 64, .context = 512, .seed = 1};
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
		case 'C':
			a.width = atoi(optarg);
			break;
		case 'T':
			a.context = atoi(optarg);
			break;
		case 's':
			a.seed = atoi(optarg);
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

static void printstats(const char *name, const float *p, int n) {
	double sum = 0, sumsq = 0;
	for (int i = 0; i < n; i++) {
		sum += p[i];
		sumsq += p[i] * p[i];
	}
	double mean = sum / n;
	printf("%s: mean %+.5f, std %.5f\n", name, mean, sqrt(sumsq / n - mean * mean));
}

// forward builds a model, embeds the start of the first sequence and puts it through the final LayerNorm
static int forward(int argc, char **argv) {
	Args a = parse(argc, argv, "m:f:C:T:s:");
	if (a.width < 1 || a.context < 1) errx(1, "-C and -T must be at least 1");
	Corpus c = corpus_load(a.path, &a.fold);
	corpus_report(&c, &a.fold, stderr);

	Config cfg = {.V = c.nvocab, .C = a.width, .T = a.context};
	Model m = model_init(cfg, a.seed);

	printf("model: V %d, C %d, T %d, seed %d\n", cfg.V, cfg.C, cfg.T, a.seed);
	printf("params: %d (wte %d, wpe %d, lnf %d)\n", m.nparams, cfg.V * cfg.C, cfg.T * cfg.C, 2 * cfg.C);
	printstats("wte", m.wte, cfg.V * cfg.C);
	printstats("wpe", m.wpe, cfg.T * cfg.C);

	const int example = 11; // the words in LEARN.md §5's example
	int n = c.start[1] - c.start[0];
	if (n > example) n = example;
	if (n > cfg.T) n = cfg.T;

	float *x = calloc((size_t)n * cfg.C, sizeof *x);
	if (x == NULL) err(1, "calloc");
	float *y = calloc((size_t)n * cfg.C, sizeof *y);
	if (y == NULL) err(1, "calloc");

	model_embed(&m, c.ids + c.start[0], n, x);
	layernorm(y, x, m.lnfw, m.lnfb, n, cfg.C);

	printf("\nthe first sequence, embedded, then through the final LayerNorm\n");
	printf("  t  word          |x|   after LayerNorm: the first numbers     mean     std    |y|\n");
	for (int t = 0; t < n; t++) {
		const float *u = x + t * cfg.C;
		const float *v = y + t * cfg.C;
		double xlen = 0, ysum = 0, ylen = 0;
		for (int i = 0; i < cfg.C; i++) {
			xlen += u[i] * u[i];
			ysum += v[i];
			ylen += v[i] * v[i];
		}
		double ym = ysum / cfg.C;
		printf("%3d  %-12s %.3f  ", t, c.vocab[c.ids[c.start[0] + t]], sqrt(xlen));
		for (int i = 0; i < 4 && i < cfg.C; i++) printf(" %+.4f", v[i]);
		printf("   %+.4f  %.4f  %.3f\n", ym, sqrt(ylen / cfg.C - ym * ym), sqrt(ylen));
	}
	free(y);
	free(x);
	return 0;
}

int main(int argc, char **argv) {
	if (argc < 2) usage();
	if (strcmp(argv[1], "corpus") == 0) return corpus(argc - 1, argv + 1);
	if (strcmp(argv[1], "bigram") == 0) return bigram(argc - 1, argv + 1);
	if (strcmp(argv[1], "forward") == 0) return forward(argc - 1, argv + 1);
	usage();
}
