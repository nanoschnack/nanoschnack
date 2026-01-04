TOKENIZER_DIR := tokenizer
TOKENIZER_INPUT := data/german.txt
TOKENIZER_OUTPUT := $(TOKENIZER_DIR)/tokenizer-v2.json
TOKENIZER_SIZE := 32000
TOKENIZER_TOP := 50
CORPUS_SIZE := 1000000000

.PHONY: tokenizer
.PHONY: all
.PHONY: train
.PHONY: train-local
.PHONY: post-train
.PHONY: chat
.PHONY: corpus

tokenizer: $(TOKENIZER_OUTPUT)

$(TOKENIZER_OUTPUT): $(TOKENIZER_INPUT)
	cat $(TOKENIZER_INPUT) | (cd $(TOKENIZER_DIR) && go run . --target $(TOKENIZER_SIZE) -f tokenizer-v2.json --top $(TOKENIZER_TOP))

corpus: $(TOKENIZER_INPUT)

$(TOKENIZER_INPUT): build_tokenizer_corpus.py
	mkdir -p $(dir $(TOKENIZER_INPUT))
	python build_tokenizer_corpus.py --size $(CORPUS_SIZE) --output $(TOKENIZER_INPUT)

all: $(TOKENIZER_OUTPUT)

train: $(TOKENIZER_OUTPUT)
	python model/training.py

train-local: $(TOKENIZER_OUTPUT)
	EMBED_SIZE=384 CONTEXT_LEN=128 NUM_LAYERS=3 BATCH_SIZE=32 python model/training.py

post-train: $(TOKENIZER_OUTPUT)
	LEARNING_RATE=6e-5 WARMUP_PCT=0.03 LEARNING_RATE_MIN_RATIO=1.0 DECAY=0.001 FREEZE_EMBEDDINGS=1 \
	  python model/training.py

chat:
	python model/chat.py
