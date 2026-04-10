# Go Tokenizer

## CLI
- Train from stdin and encode the last non-empty line.
- Output token IDs to stdout or a file.

Examples:
```
cat german.txt | go run . --target 32000 -f tokenizer-v3.json
cat german.txt | go run . --target 32000 -f tokenizer-v3.json --in "Das ist ein Test."
cat german.txt | go run . --target 32000 --top 50
cat german.txt | go run . --target 32000
```

Notes:
- All non-empty lines are used for training.
- `--top` prints the longest tokens by decoded byte length with corpus counts.
- Keep legacy artifacts like `tokenizer.json` unchanged for older checkpoints. Write
  regenerated UTF-8-safe artifacts to `tokenizer-v3.json` for new training runs.
