from tokenizers import Tokenizer

from model import setup_paths

setup_paths()

from chat import _build_byte_decoder


def _decode_token(token):
    # Decode byte-level token strings back into readable UTF-8 text.
    byte_decoder = _build_byte_decoder()
    payload = bytes(byte_decoder[ch] for ch in token)
    return payload.decode("utf-8", errors="replace")


tok = Tokenizer.from_file("tokenizer/tokenizer.json")

samples = [
    "für München ist Käse schön",
    "größer, älter, früher, übermäßig",
    "Frühstück in Österreich",
]

for s in samples:
    enc = tok.encode(s)
    print()
    print(s)
    print([_decode_token(token) for token in enc.tokens])
