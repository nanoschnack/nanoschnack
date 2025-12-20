package tokenizer

import (
	"reflect"
	"testing"
)

// helper to build a simple iterator over a slice.
func sliceIter(items []string) func() (string, bool) {
	i := 0
	return func() (string, bool) {
		if i >= len(items) {
			return "", false
		}
		v := items[i]
		i++
		return v, true
	}
}

// inspired by https://github.com/karpathy/minbpe?tab=readme-ov-file#quick-start
//
//from minbpe import BasicTokenizer
//tokenizer = BasicTokenizer()
//text = "aaabdaaabac"
//tokenizer.train(text, 256 + 3) # 256 are the byte tokens, then do 3 merges
//print(tokenizer.encode(text))
//# [258, 100, 258, 97, 99]
//print(tokenizer.decode([258, 100, 258, 97, 99]))
//# aaabdaaabac
//tokenizer.save("toy")
//# writes two files: toy.model (for loading) and toy.vocab (for viewing)
func TestTrainAndEncodeSimple(t *testing.T) {
	tok := NewTokenizer()
	iter := sliceIter([]string{"aaabdaaabac"})

	if err := tok.TrainFromIterator(iter, 256+3, 16, ".+"); err != nil {
		t.Fatalf("train failed: %v", err)
	}

	if got := len(tok.Merges); got != 3 {
		t.Fatalf("expected 3 merges, got %d", got)
	}

	got, err := tok.Encode("aaabdaaabac")
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	want := []uint32{258, 100, 258, 97, 99}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("want %v, got %v", want, got)
	}
}

func TestMergeableRanksIncludesBaseAndMerges(t *testing.T) {
	tok := NewTokenizer()
	iter := sliceIter([]string{"aa"})

	if err := tok.TrainFromIterator(iter, 257, 4, ".+"); err != nil {
		t.Fatalf("train failed: %v", err)
	}

	ranks := tok.GetMergeableRanks()
	expected := 257 // 256 base bytes + 1 merge
	if len(ranks) != expected {
		t.Fatalf("expected %d mergeable ranks, got %d", expected, len(ranks))
	}

	if tok.GetPattern() != ".+" {
		t.Fatalf("pattern not retained; expected '.+', got %q", tok.GetPattern())
	}
}
