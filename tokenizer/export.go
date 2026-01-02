package main

import (
	"encoding/json"
	"os"
	"sort"
	"strings"
)

func WriteTokenizerJSON(path string, t *Tokenizer) error {
	encoder := buildByteEncoder()
	mergeable := t.GetMergeableRanks()
	vocab := make(map[string]uint32, len(mergeable))
	idToToken := make(map[uint32]string, len(mergeable))
	for _, entry := range mergeable {
		token := encodeTokenBytes(encoder, entry.Bytes)
		vocab[token] = entry.ID
		idToToken[entry.ID] = token
	}

	type mergeEntry struct {
		pair Pair
		id   uint32
	}
	entries := make([]mergeEntry, 0, len(t.Merges))
	for pair, id := range t.Merges {
		entries = append(entries, mergeEntry{pair: pair, id: id})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].id < entries[j].id
	})

	merges := make([]string, 0, len(entries))
	for _, entry := range entries {
		left := idToToken[entry.pair.A]
		right := idToToken[entry.pair.B]
		merges = append(merges, left+" "+right)
	}

	// Ensure the regex is JS-compatible for Transformers.js.
	pattern := jsCompatiblePattern(t.GetPattern())

	preTokenizer := map[string]any{
		"type": "Sequence",
		"pretokenizers": []any{
			map[string]any{
				"type":     "Split",
				"pattern":  map[string]any{"Regex": pattern},
				"behavior": "Isolated",
				"invert":   false,
			},
			map[string]any{
				"type":             "ByteLevel",
				"add_prefix_space": false,
				"trim_offsets":     false,
				"use_regex":        false,
			},
		},
	}

	model := map[string]any{
		"type":                     "BPE",
		"dropout":                  nil,
		"unk_token":                nil,
		"continuing_subword_prefix": "",
		"end_of_word_suffix":       "",
		"vocab":                    vocab,
		"merges":                   merges,
		"fuse_unk":                 false,
		"byte_fallback":            false,
	}

	tokenizerJSON := map[string]any{
		"version":        "1.0",
		"truncation":     nil,
		"padding":        nil,
		"added_tokens":   []any{},
		"normalizer":     nil,
		"pre_tokenizer":  preTokenizer,
		"post_processor": nil,
		"decoder": map[string]any{
			"type":             "ByteLevel",
			"add_prefix_space": false,
			"trim_offsets":     false,
		},
		"model": model,
	}

	encoded, err := json.MarshalIndent(tokenizerJSON, "", "  ")
	if err != nil {
		return err
	}
	encoded = append(encoded, '\n')
	return os.WriteFile(path, encoded, 0o644)
}

func buildByteEncoder() [256]rune {
	var bs []int
	for i := 33; i <= 126; i++ {
		bs = append(bs, i)
	}
	for i := 161; i <= 172; i++ {
		bs = append(bs, i)
	}
	for i := 174; i <= 255; i++ {
		bs = append(bs, i)
	}

	var used [256]bool
	for _, b := range bs {
		used[b] = true
	}

	cs := make([]int, len(bs))
	copy(cs, bs)
	n := 0
	for b := 0; b < 256; b++ {
		if !used[b] {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}

	var encoder [256]rune
	for i, b := range bs {
		encoder[byte(b)] = rune(cs[i])
	}
	return encoder
}

func encodeTokenBytes(encoder [256]rune, data []byte) string {
	var b strings.Builder
	b.Grow(len(data))
	for _, v := range data {
		b.WriteRune(encoder[v])
	}
	return b.String()
}

func jsCompatiblePattern(pattern string) string {
	replacer := strings.NewReplacer(
		"(?>", "(?:",
		"'(?i:[sdmt]|ll|ve|re)", "(?i:'s|'t|'re|'ve|'m|'ll|'d)",
	)
	return replacer.Replace(pattern)
}
