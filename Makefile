tok: gpt
	./gpt corpus -m 5 -f SK_,NPC_ data/corpus.txt

bigram: gpt
	./gpt bigram -m 5 -f SK_,NPC_ data/corpus.txt

forward: gpt
	./gpt forward -m 5 -f SK_,NPC_ data/corpus.txt

gpt: main.c tok.c tok.h bigram.c bigram.h model.c model.h
	cc -O2 -Wall -Wextra -std=c17 -o gpt main.c tok.c bigram.c model.c -lm

.PHONY: tok bigram forward
