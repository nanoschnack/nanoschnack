# Go Tokenizer

## CLI
- Train from stdin and encode the last non-empty line.
- Output token IDs to stdout or a file.

Examples:
```
cat icelandic-1b.txt | go run . --target 32000 -f tokenizer-icelandic-v1.json
cat icelandic-1b.txt | go run . --target 32000 -f tokenizer-icelandic-v1.json --in "Þetta er prófun."
cat icelandic-1b.txt | go run . --target 32000 --top 50
cat icelandic-1b.txt | go run . --target 32000
```

Notes:
- All non-empty lines are used for training.
- `--top` prints the longest tokens by decoded byte length with corpus counts.
- Keep legacy artifacts through `tokenizer-v3.json` unchanged for older checkpoints. The
  Icelandic corpus is written to the UTF-8-safe `tokenizer-icelandic-v1.json` artifact.
