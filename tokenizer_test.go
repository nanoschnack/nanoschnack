package tokenizer

import "testing"

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

func TestTrainAndEncodeSimple(t *testing.T) {
	tok := NewTokenizer()
	iter := sliceIter([]string{"aa"})

	if err := tok.TrainFromIterator(iter, 257, 16, ".+"); err != nil {
		t.Fatalf("train failed: %v", err)
	}

	if got := len(tok.Merges); got != 1 {
		t.Fatalf("expected 1 merge, got %d", got)
	}

	encoded, err := tok.Encode("aa")
	if err != nil {
		t.Fatalf("encode failed: %v", err)
	}

	if len(encoded) != 1 || encoded[0] != 256 {
		t.Fatalf("expected single merged token 256, got %v", encoded)
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
