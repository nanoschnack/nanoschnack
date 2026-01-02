package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
)

func main() {
	target := flag.Uint("target", 32000, "target vocabulary size")
	jsonPath := flag.String("f", "", "output Hugging Face tokenizer.json")
	inText := flag.String("in", "", "input text to encode and print token ids")
	flag.Parse()

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)

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
