package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

func main() {
	target := flag.Uint("target", 32000, "target vocabulary size")
	jsonPath := flag.String("f", "", "output Hugging Face tokenizer.json")
	inText := flag.String("in", "", "input text to encode and print token ids")
	topN := flag.Int("top", 0, "print N longest tokens with decoded text and counts")
	flag.Parse()

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)

	log.Printf("Loading corpus...")
	var lines []string
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}

	var data []string
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		data = append(data, line)
	}
	if len(data) == 0 {
		fmt.Fprintln(os.Stderr, "no non-empty lines on stdin")
		os.Exit(1)
	}

	trainLines := data

	log.Printf("Training tokenizer: lines=%d target_vocab=%d", len(trainLines), *target)
	iter := func() func() (string, bool) {
		i := 0
		return func() (string, bool) {
			if i >= len(trainLines) {
				return "", false
			}
			s := trainLines[i]
			i++
			return s, true
		}
	}()

	t := NewTokenizer()
	if err := t.TrainFromIterator(iter, uint32(*target), 1024, GPT4Pattern); err != nil {
		panic(err)
	}

	if *topN > 0 {
		counts := countTokenFrequencies(t, trainLines)
		printTopTokens(t, *topN, counts)
	}

	if *jsonPath != "" {
		if err := WriteTokenizerJSON(*jsonPath, t); err != nil {
			panic(err)
		}
	}

	if *inText != "" {
		ids, err := t.Encode(*inText)
		if err != nil {
			panic(err)
		}

		var b strings.Builder
		for i, id := range ids {
			if i > 0 {
				b.WriteByte(' ')
			}
			fmt.Fprint(&b, id)
		}
		b.WriteByte('\n')
		fmt.Print(b.String())
	}
}

func printTopTokens(t *Tokenizer, topN int, counts map[uint32]int64) {
	mergeable := t.GetMergeableRanks()
	if topN > len(mergeable) {
		topN = len(mergeable)
	}
	sort.SliceStable(mergeable, func(i, j int) bool {
		left := len(mergeable[i].Bytes)
		right := len(mergeable[j].Bytes)
		if left == right {
			return mergeable[i].ID < mergeable[j].ID
		}
		return left > right
	})
	for i := 0; i < topN; i++ {
		entry := mergeable[i]
		fmt.Printf("%d: %s (count=%d)\n", entry.ID, renderToken(entry.Bytes), counts[entry.ID])
	}
}

func renderToken(data []byte) string {
	if utf8.Valid(data) {
		value := string(data)
		printable := true
		for _, r := range value {
			if !unicode.IsPrint(r) {
				printable = false
				break
			}
		}
		if printable {
			return value
		}
	}
	return strconv.QuoteToASCII(string(data))
}

func countTokenFrequencies(t *Tokenizer, lines []string) map[uint32]int64 {
	log.Printf("Counting token frequencies...")
	counts := make(map[uint32]int64)
	lastLogTime := time.Now().Add(-2 * time.Second)
	for i, line := range lines {
		ids, err := t.Encode(line)
		if err != nil {
			panic(err)
		}
		for _, id := range ids {
			counts[id]++
		}
		if time.Since(lastLogTime) >= 2*time.Second {
			log.Printf("Frequency progress: lines=%d", i+1)
			lastLogTime = time.Now()
		}
	}
	return counts
}
