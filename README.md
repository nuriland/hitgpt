# hitgpt

A GPT trained from scratch to predict AION 2 fights. Work in progress. Requires quite a bit of fight logs

## Use

```bash
> a2k show -json fight.pcap | python3 prep.py >> data/corpus.txt
> make
```

Needs Python 3.12+ and a C compiler.