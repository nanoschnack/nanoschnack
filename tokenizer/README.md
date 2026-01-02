# Go Tokenizer

## CLI
- Train from stdin and encode the last non-empty line.
- Output token IDs to stdout or a file.

Examples:
```
cat german.txt | go run . --target 32000 -f encoded.txt
cat german.txt | go run . --target 32000
```

Notes:
- All non-empty lines except the last are used for training.
- If stdin has a single non-empty line, it is used for both training and encoding.
