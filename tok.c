#include "tok.h"

#include <err.h>
#include <stdlib.h>
#include <string.h>

enum {
	MAXVOCAB = 1 << 16,
	MAXWORD = 1 << 17,  // distinct words before folding
	NSLOT = 1 << 18,
};

typedef struct {
	char *word[MAXWORD];
	int slot[NSLOT];
	int n;
} Words;

typedef struct {
	char *word;
	int count, index;
} Entry;

static void *xcalloc(size_t n, size_t size) {
	void *p = calloc(n, size);
	if (p == NULL && n > 0 && size > 0) err(1, "calloc");
	return p;
}

static char *slurp(const char *path) {
	FILE *f = fopen(path, "rb");
	if (f == NULL) err(1, "%s", path);

	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	if (n < 0) err(1, "%s", path);

	rewind(f);
	char *s = xcalloc(n + 1, 1);
	if (fread(s, 1, n, f) != (size_t)n) errx(1, "%s: short read", path);

	fclose(f);
	return s;
}

// fnv hashes a word to a number, so intern can jump straight to its slot in the table
// (FNV-1a: for each byte, XOR it in, then multiply by the FNV prime to spread its bits over all 32).
// Honestly just look at https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
static uint32_t fnv(const char *s) {
	uint32_t h = 2166136261u;
	for (; *s != 0; s++) {
		h ^= (uint8_t)*s;
		h *= 16777619u;
	}
	return h;
}

// intern returns w's index, adding w if it's new
static int intern(Words *t, char *w) {
	for (uint32_t h = fnv(w);; h++) {
		int *s = &t->slot[h & (NSLOT - 1)];
		if (*s == 0) {
			if (t->n == MAXWORD) errx(1, "more than %d distinct words", MAXWORD);
			t->word[t->n] = w;
			*s = t->n + 1;
			return t->n++;
		}
		if (strcmp(t->word[*s - 1], w) == 0) return *s - 1;
	}
}

static int prefixed(const char *w, const Fold *fold) {
	for (int k = 0; k < fold->n; k++)
		if (strncmp(w, fold->prefix[k], strlen(fold->prefix[k])) == 0) return k;
	return -1;
}

static int issep(char ch) { return ch == ' ' || ch == '\n' || ch == 0; }

static int bycount(const void *a, const void *b) {
	const Entry *x = a, *y = b;
	if (x->count != y->count) return y->count - x->count;
	return strcmp(x->word, y->word);
}

static int byint(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

// cut cuts text into words in place, and returns each word's index in t, in the text's order.
static int *cut(char *text, Words *t, Corpus *c) {
	int nword = 0, nline = 1;
	for (char *p = text; *p != 0; p++) {
		if (!issep(p[0]) && issep(p[1])) nword++;
		if (*p == '\n') nline++;
	}

	int *w = xcalloc(nword, sizeof *w);
	c->start = xcalloc(nline + 1, sizeof *c->start);
	for (char *p = text;; p++) {
		char *word = p;
		while (!issep(*p)) p++;

		char sep = *p;
		*p = 0;
		if (p > word) w[c->nids++] = intern(t, word);

		int lineended = sep != ' ';
		if (lineended && c->nids > c->start[c->nseq]) {
			c->nseq++;
			c->start[c->nseq] = c->nids;
		}
		if (sep == 0) break;
	}
	return w;
}

// foldinto returns what each word in t folds into, either itself, or its prefix's "?" word.
//  It counts the words folded under each prefix in folded.
static int *foldinto(Words *t, const int *w, int nw, const Fold *fold, int *folded) {
	int target[MAXFOLD];
	for (int k = 0; k < fold->n; k++) {
		size_t n = strlen(fold->prefix[k]) + 2;
		char *q = xcalloc(n, 1);

		snprintf(q, n, "%s?", fold->prefix[k]);
		target[k] = intern(t, q);
		if (t->word[target[k]] != q) free(q);
	}

	int *seen = xcalloc(t->n, sizeof *seen);
	for (int i = 0; i < nw; i++) seen[w[i]]++;

	int *into = xcalloc(t->n, sizeof *into);
	for (int i = 0; i < t->n; i++) {
		into[i] = i;
		if (seen[i] >= fold->min || strchr(t->word[i], '?') != NULL) continue;

		int k = prefixed(t->word[i], fold);
		if (k < 0) continue;

		into[i] = target[k];
		folded[k]++;
	}
	free(seen);
	return into;
}

// number gives each word left after folding an ID
static void number(Corpus *c, const Words *t, const int *w, const int *into) {
	int *total = xcalloc(t->n, sizeof *total);
	for (int i = 0; i < c->nids; i++) total[into[w[i]]]++;

	Entry *e = xcalloc(t->n, sizeof *e);
	for (int i = 0; i < t->n; i++)
		if (total[i] > 0) e[c->nvocab++] = (Entry){t->word[i], total[i], i};
	if (c->nvocab > MAXVOCAB) errx(1, "a vocab of %d words; IDs are uint16, so %d at most", c->nvocab, MAXVOCAB);
	qsort(e, c->nvocab, sizeof *e, bycount);

	// Allocate memory for the ID array and the vocabulary arrays
	int *id = xcalloc(t->n, sizeof *id);
	c->vocab = xcalloc(c->nvocab, sizeof *c->vocab);
	c->count = xcalloc(c->nvocab, sizeof *c->count);

	// Map each word to its ID, and store the word and its count
	for (int v = 0; v < c->nvocab; v++) {
		id[e[v].index] = v;
		c->vocab[v] = e[v].word;
		c->count[v] = e[v].count;
	}

	c->ids = xcalloc(c->nids, sizeof *c->ids);
	for (int i = 0; i < c->nids; i++) c->ids[i] = id[into[w[i]]];

	free(total);
	free(e);
	free(id);
}

// corpus_load loads the corpus from a file and folds it according to the given fold configuration
Corpus corpus_load(const char *path, const Fold *fold) {
	Corpus c = {0};
	Words *t = xcalloc(1, sizeof *t);
	char *text = slurp(path); // the words point into it, so it's never freed

	int *w = cut(text, t, &c);
	if (c.nids == 0) errx(1, "%s: no words", path);

	c.unfolded = t->n;
	int *into = foldinto(t, w, c.nids, fold, c.folded);
	number(&c, t, w, into);

	free(t);
	free(w);
	free(into);
	return c;
}

void corpus_report(const Corpus *c, const Fold *fold, FILE *out) {
	int folded = 0;
	for (int k = 0; k < fold->n; k++) folded += c->folded[k];

	fprintf(out, "corpus: %d sequences, %d words, vocab %d", c->nseq, c->nids, c->unfolded);
	if (folded > 0) {
		fprintf(out, ", %d after folding %d rare word%s (", c->nvocab, folded, folded == 1 ? "" : "s");
		const char *sep = "";
		for (int k = 0; k < fold->n; k++) {
			if (c->folded[k] == 0) continue;
			fprintf(out, "%s%d %s", sep, c->folded[k], fold->prefix[k]);
			sep = ", ";
		}
		fputc(')', out);
	}
	fputc('\n', out);

	int *len = xcalloc(c->nseq, sizeof *len);
	int nval = 0;
	for (int s = 0; s < c->nseq; s++) {
		len[s] = c->start[s + 1] - c->start[s];
		if (s % VALEVERY == VALEVERY - 1) nval += len[s];
	}
	qsort(len, c->nseq, sizeof *len, byint);

	int median = len[c->nseq / 2];
	int p95 = len[(95 * c->nseq + 99) / 100 - 1]; // the nearest rank, i.e. ceil(0.95 n), counted from 1
	fprintf(out, "sequences: %d to %d words, median %d, p95 %d: the context should be at least %d\n",
		len[0], len[c->nseq - 1], median, p95, p95);

	int valseq = c->nseq / VALEVERY;
	fprintf(out, "split: train %d sequences, %d words; val %d sequences, %d words\n",
		c->nseq - valseq, c->nids - nval, valseq, nval);
	free(len);
}
