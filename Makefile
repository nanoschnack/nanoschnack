TOKENIZER_DIR := tokenizer
TOKENIZER_INPUT := data/german.txt
TOKENIZER_OUTPUT := tokenizer.json
TOKENIZER_SIZE := 2000
CORPUS_SIZE := 1000000000

.PHONY: tokenizer
.PHONY: all
.PHONY: train
.PHONY: chat
.PHONY: corpus

tokenizer: $(TOKENIZER_OUTPUT)

$(TOKENIZER_OUTPUT): $(TOKENIZER_INPUT)
	cat $(TOKENIZER_INPUT) | go run $(TOKENIZER_DIR) --target $(TOKENIZER_SIZE) -f $(TOKENIZER_OUTPUT)

corpus: $(TOKENIZER_INPUT)

$(TOKENIZER_INPUT): build_tokenizer_corpus.py
	mkdir -p $(dir $(TOKENIZER_INPUT))
	python build_tokenizer_corpus.py --size $(CORPUS_SIZE) --output $(TOKENIZER_INPUT)

all: $(TOKENIZER_OUTPUT)

train: $(TOKENIZER_OUTPUT)
	EMBED_SIZE=384 CONTEXT_LEN=128 NUM_LAYERS=3 BATCH_SIZE=32 python model/training.py

chat:
	python model/chat.py
