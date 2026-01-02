package main

import (
	"container/heap"
	"errors"
	"log"
	"runtime"
	"sort"
	"sync"

	"github.com/dlclark/regexp2"
)

// Default GPT-4 style regex pattern for splitting text.
// Note: regexp2 (Go/.NET syntax) does not support possessive quantifiers, so we
// use atomic groups to approximate the PCRE-style pattern used elsewhere.
//
// Here’s what GPT4Pattern is doing (left-to-right alternatives, first match wins):
//
//  - '(?i:[sdmt]|ll|ve|re): contractions starting with an apostrophe followed by s/d/m/t or the suffixes ll/ve/re (case-insensitive inside the group).
//  - (?>[^\r\n\p{L}\p{N}]?)\p{L}+: a word: optional leading non-letter/number (e.g., a leading period in “.word”), then one or more Unicode letters. The atomic group (?>) prevents
//    backtracking.
//  - \p{N}{1,3}: a number chunk of 1–3 Unicode digits (splits long numbers into 3-digit pieces).
//  -  ?(?>[^\s\p{L}\p{N}]+)[\r\n]*: optional leading space, then a run of symbols/punctuation (no letters/numbers/whitespace), then optional trailing newlines—captures things like " --"
//    or " ##\n".
//  - \s*[\r\n]: optional whitespace followed by a newline (CR or LF) to ensure newlines are isolated.
//  - \s+(?!\S): trailing whitespace at end of string (whitespace not followed by a non-space).
//  - \s+: any remaining whitespace runs.
const GPT4Pattern = `'(?i:[sdmt]|ll|ve|re)|(?>[^\r\n\p{L}\p{N}]?)\p{L}+|\p{N}{1,3}| ?(?>[^\s\p{L}\p{N}]+)[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+`

type Pair struct {
	A uint32
	B uint32
}

// Tokenizer is a Byte Pair Encoding tokenizer similar to the Rust implementation.
type Tokenizer struct {
	Merges          map[Pair]uint32
	pattern         string
	compiledPattern *regexp2.Regexp
}

// Word represents a tokenized chunk as a sequence of ids.
type Word struct {
	ids []uint32
}

type pairDelta struct {
	pair  Pair
	delta int32
}

// MergeJob represents a candidate pair merge in the priority queue.
type MergeJob struct {
	pair  Pair
	count int64
	pos   map[int]struct{}
}

type mergeHeap []*MergeJob

// NewTokenizer constructs an empty tokenizer.
func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		Merges:          make(map[Pair]uint32),
		pattern:         "",
		compiledPattern: nil,
	}
}

// pairs yields all consecutive token pairs within the word.
func (w *Word) pairs() []Pair {
	if len(w.ids) < 2 {
		return nil
	}
	out := make([]Pair, 0, len(w.ids)-1)
	for i := 0; i < len(w.ids)-1; i++ {
		out = append(out, Pair{w.ids[i], w.ids[i+1]})
	}
	return out
}

// mergePair merges all non-overlapping occurrences of pair -> newID within the word.
// It returns the pair-count deltas for this word only.
func (w *Word) mergePair(pair Pair, newID uint32) []pairDelta {
	a, b := pair.A, pair.B
	n := len(w.ids)
	if n < 2 {
		return nil
	}

	out := make([]uint32, 0, n)
	deltas := make([]pairDelta, 0, 6)

	for i := 0; i < n; {
		if i+1 < n && w.ids[i] == a && w.ids[i+1] == b {
			var left *uint32
			if len(out) > 0 {
				v := out[len(out)-1]
				left = &v
			}
			var right *uint32
			if i+2 < n {
				v := w.ids[i+2]
				right = &v
			}

			if left != nil {
				deltas = append(deltas, pairDelta{pair: Pair{*left, a}, delta: -1})
				deltas = append(deltas, pairDelta{pair: Pair{*left, newID}, delta: 1})
			}
			deltas = append(deltas, pairDelta{pair: Pair{a, b}, delta: -1})
			if right != nil {
				deltas = append(deltas, pairDelta{pair: Pair{b, *right}, delta: -1})
				deltas = append(deltas, pairDelta{pair: Pair{newID, *right}, delta: 1})
			}

			out = append(out, newID)
			i += 2
		} else {
			out = append(out, w.ids[i])
			i++
		}
	}

	w.ids = out
	return deltas
}

func pairLess(a, b Pair) bool {
	if a.A == b.A {
		return a.B < b.B
	}
	return a.A < b.A
}

// heap.Interface implementation
func (h mergeHeap) Len() int { return len(h) }

func (h mergeHeap) Less(i, j int) bool {
	if h[i].count == h[j].count {
		return pairLess(h[i].pair, h[j].pair)
	}
	return h[i].count > h[j].count
}

func (h mergeHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *mergeHeap) Push(x any) {
	*h = append(*h, x.(*MergeJob))
}

func (h *mergeHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func countPairs(words []Word, counts []int64) (map[Pair]int64, map[Pair]map[int]struct{}) {
	workerCount := runtime.NumCPU()
	jobs := make(chan int, workerCount)
	results := make(chan struct {
		pc  map[Pair]int64
		wtu map[Pair]map[int]struct{}
	}, workerCount)

	var wg sync.WaitGroup
	wg.Add(workerCount)
	for w := 0; w < workerCount; w++ {
		go func() {
			defer wg.Done()
			localPC := make(map[Pair]int64)
			localWTU := make(map[Pair]map[int]struct{})
			for idx := range jobs {
				word := words[idx]
				if len(word.ids) < 2 || counts[idx] == 0 {
					continue
				}
				for _, p := range word.pairs() {
					localPC[p] += counts[idx]
					if _, ok := localWTU[p]; !ok {
						localWTU[p] = make(map[int]struct{})
					}
					localWTU[p][idx] = struct{}{}
				}
			}
			results <- struct {
				pc  map[Pair]int64
				wtu map[Pair]map[int]struct{}
			}{pc: localPC, wtu: localWTU}
		}()
	}

	for i := range words {
		jobs <- i
	}
	close(jobs)
	go func() {
		wg.Wait()
		close(results)
	}()

	pairCounts := make(map[Pair]int64)
	whereToUpdate := make(map[Pair]map[int]struct{})
	for res := range results {
		for p, c := range res.pc {
			pairCounts[p] += c
		}
		for p, set := range res.wtu {
			tgt := whereToUpdate[p]
			if tgt == nil {
				tgt = make(map[int]struct{})
				whereToUpdate[p] = tgt
			}
			for idx := range set {
				tgt[idx] = struct{}{}
			}
		}
	}

	return pairCounts, whereToUpdate
}

// trainCoreIncremental performs the core BPE training loop.
func (t *Tokenizer) trainCoreIncremental(words []Word, counts []int64, vocabSize uint32) error {
	if vocabSize < 256 {
		return errors.New("vocab_size must be at least 256")
	}
	numMerges := vocabSize - 256
	t.Merges = make(map[Pair]uint32)

	pairCounts, whereToUpdate := countPairs(words, counts)

	h := mergeHeap{}
	for pair, pos := range whereToUpdate {
		if c := pairCounts[pair]; c > 0 {
			heap.Push(&h, &MergeJob{pair: pair, count: c, pos: pos})
		}
	}

	var mergesDone uint32
	var lastLogPercent uint32

	for mergesDone < numMerges {
		if h.Len() == 0 {
			break
		}
		top := heap.Pop(&h).(*MergeJob)
		current := pairCounts[top.pair]
		if top.count != current {
			top.count = current
			if top.count > 0 {
				heap.Push(&h, top)
			}
			continue
		}
		if top.count == 0 {
			break
		}

		newID := 256 + mergesDone
		t.Merges[top.pair] = newID

		localPosUpdates := make(map[Pair]map[int]struct{})
		for wordIdx := range top.pos {
			changes := words[wordIdx].mergePair(top.pair, newID)
			for _, ch := range changes {
				deltaTotal := int64(ch.delta) * counts[wordIdx]
				if deltaTotal != 0 {
					pairCounts[ch.pair] += deltaTotal
					if ch.delta > 0 {
						if _, ok := localPosUpdates[ch.pair]; !ok {
							localPosUpdates[ch.pair] = make(map[int]struct{})
						}
						localPosUpdates[ch.pair][wordIdx] = struct{}{}
					}
				}
			}
		}

		for pair, pos := range localPosUpdates {
			if cnt := pairCounts[pair]; cnt > 0 {
				heap.Push(&h, &MergeJob{pair: pair, count: cnt, pos: pos})
			}
		}

		mergesDone++

		currentPercent := (mergesDone * 100) / numMerges
		if currentPercent > lastLogPercent {
			log.Printf("Progress: %d%% (%d/%d merges) - Last merge: (%d,%d) -> %d (frequency: %d)",
				currentPercent, mergesDone, numMerges, top.pair.A, top.pair.B, newID, top.count)
			lastLogPercent = currentPercent
		}
	}

	log.Printf("Finished training: %d merges completed", mergesDone)
	return nil
}

// TrainFromIterator ingests a streaming iterator of strings and trains the tokenizer.
// The iterator should return (value, true) while data remains, and (_, false) when exhausted.
func (t *Tokenizer) TrainFromIterator(iter func() (string, bool), vocabSize uint32, bufferSize int, pattern string) error {
	if bufferSize <= 0 {
		bufferSize = 8192
	}

	patternStr := pattern
	if patternStr == "" {
		patternStr = GPT4Pattern
	}

	compiled, err := regexp2.Compile(patternStr, 0)
	if err != nil {
		return err
	}

	t.pattern = patternStr
	t.compiledPattern = compiled

	counts := make(map[string]int64)
	buf := make([]string, 0, bufferSize)
	exhausted := false

	for {
		buf = buf[:0]
		for len(buf) < bufferSize {
			next, ok := iter()
			if !ok {
				exhausted = true
				break
			}
			buf = append(buf, next)
		}

		if len(buf) == 0 && exhausted {
			break
		}

		local, err := t.countBufferParallel(buf)
		if err != nil {
			return err
		}
		for k, v := range local {
			counts[k] += v
		}

		if exhausted {
			break
		}
	}

	words := make([]Word, 0, len(counts))
	cvec := make([]int64, 0, len(counts))
	for chunk, c := range counts {
		ids := make([]uint32, len(chunk))
		for i := range chunk {
			ids[i] = uint32(chunk[i])
		}
		words = append(words, Word{ids: ids})
		cvec = append(cvec, c)
	}

	return t.trainCoreIncremental(words, cvec, vocabSize)
}

// GetPattern returns the regex pattern being used.
func (t *Tokenizer) GetPattern() string {
	return t.pattern
}

// countBufferParallel counts regex matches across the buffer using worker goroutines.
func (t *Tokenizer) countBufferParallel(buf []string) (map[string]int64, error) {
	workerCount := runtime.NumCPU()
	jobs := make(chan string, workerCount)
	results := make(chan map[string]int64, workerCount)
	errCh := make(chan error, 1)

	var wg sync.WaitGroup
	wg.Add(workerCount)
	for w := 0; w < workerCount; w++ {
		go func() {
			defer wg.Done()
			local := make(map[string]int64)
			for s := range jobs {
				match, err := t.compiledPattern.FindStringMatch(s)
				if err != nil {
					select {
					case errCh <- err:
					default:
					}
					break
				}
				for match != nil {
					local[match.String()]++
					match, err = t.compiledPattern.FindNextMatch(match)
					if err != nil {
						select {
						case errCh <- err:
						default:
						}
						match = nil
						break
					}
				}
			}
			results <- local
		}()
	}

	for _, s := range buf {
		jobs <- s
	}
	close(jobs)

	go func() {
		wg.Wait()
		close(results)
	}()

	agg := make(map[string]int64)
	for local := range results {
		for k, v := range local {
			agg[k] += v
		}
	}

	select {
	case err := <-errCh:
		return nil, err
	default:
	}

	return agg, nil
}

// MergeableRank describes a token's byte sequence and id.
type MergeableRank struct {
	Bytes []byte
	ID    uint32
}

// GetMergeableRanks returns the mergeable ranks (token bytes -> token id / rank).
func (t *Tokenizer) GetMergeableRanks() []MergeableRank {
	mergeable := make([]MergeableRank, 0, len(t.Merges)+256)

	tokenBytes := make([][]byte, 256)
	for i := 0; i < 256; i++ {
		tokenBytes[i] = []byte{byte(i)}
		mergeable = append(mergeable, MergeableRank{Bytes: append([]byte(nil), byte(i)), ID: uint32(i)})
	}

	sorted := make([]struct {
		pair Pair
		id   uint32
	}, 0, len(t.Merges))
	for p, id := range t.Merges {
		sorted = append(sorted, struct {
			pair Pair
			id   uint32
		}{pair: p, id: id})
	}

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].id < sorted[j].id
	})

	for _, entry := range sorted {
		left, right := entry.pair.A, entry.pair.B
		merged := append(append([]byte(nil), tokenBytes[left]...), tokenBytes[right]...)
		if int(entry.id) >= len(tokenBytes) {
			tmp := make([][]byte, int(entry.id)+1)
			copy(tmp, tokenBytes)
			tokenBytes = tmp
		}
		tokenBytes[entry.id] = merged
		mergeable = append(mergeable, MergeableRank{Bytes: append([]byte(nil), merged...), ID: entry.id})
	}

	return mergeable
}

// Encode converts the input text into token IDs using the learned merges.
func (t *Tokenizer) Encode(text string) ([]uint32, error) {
	if t.compiledPattern == nil {
		return nil, errors.New("tokenizer is not initialized")
	}

	var all []uint32

	match, err := t.compiledPattern.FindStringMatch(text)
	if err != nil {
		return nil, err
	}
	for match != nil {
		chunk := match.String()
		ids := make([]uint32, len(chunk))
		for i := range chunk {
			ids[i] = uint32(chunk[i])
		}

		for len(ids) >= 2 {
			found := false
			var bestIdx int
			var bestID uint32
			for i := 0; i < len(ids)-1; i++ {
				pair := Pair{ids[i], ids[i+1]}
				if newID, ok := t.Merges[pair]; ok {
					if !found || newID < bestID {
						bestIdx = i
						bestID = newID
						found = true
					}
				}
			}
			if !found {
				break
			}
			ids[bestIdx] = bestID
			ids = append(ids[:bestIdx+1], ids[bestIdx+2:]...)
		}

		all = append(all, ids...)

		match, err = t.compiledPattern.FindNextMatch(match)
		if err != nil {
			return nil, err
		}
	}

	return all, nil
}
