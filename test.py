class SimpleTokenizer:
    """
    A very basic tokenizer that:
    1. Converts text to lowercase
    2. Splits text into words
    """

    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()


class Vocabulary:
    """
    Builds a vocabulary from tokenized sentences and
    converts tokens <-> IDs.
    """

    def __init__(self):

        # Special Tokens
        self.token_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }

        self.id_to_token = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<BOS>",
            3: "<EOS>",
        }

    def build(self, tokenized_sentences):

        next_id = len(self.token_to_id)

        for sentence in tokenized_sentences:

            for token in sentence:

                if token not in self.token_to_id:

                    self.token_to_id[token] = next_id
                    self.id_to_token[next_id] = token

                    next_id += 1

    def encode(self, tokens):

        ids = [self.token_to_id["<BOS>"]]

        for token in tokens:

            ids.append(
                self.token_to_id.get(
                    token,
                    self.token_to_id["<UNK>"]
                )
            )

        ids.append(self.token_to_id["<EOS>"])

        return ids

    def decode(self, ids):

        tokens = []

        for idx in ids:

            tokens.append(
                self.id_to_token.get(
                    idx,
                    "<UNK>"
                )
            )

        return tokens


def main():

    # Dataset
    sentences = [
        "I love AI",
        "I love Python",
        "Transformers are amazing"
    ]

    # -------------------------
    # Tokenization
    # -------------------------

    tokenizer = SimpleTokenizer()

    tokenized_sentences = []

    for sentence in sentences:

        tokens = tokenizer.tokenize(sentence)

        tokenized_sentences.append(tokens)

    print("=" * 50)
    print("Tokenized Sentences")
    print("=" * 50)

    for tokens in tokenized_sentences:
        print(tokens)

    # -------------------------
    # Vocabulary
    # -------------------------

    vocab = Vocabulary()

    vocab.build(tokenized_sentences)

    print("\n" + "=" * 50)
    print("Vocabulary")
    print("=" * 50)

    for token, idx in vocab.token_to_id.items():
        print(f"{token:15} --> {idx}")

    # -------------------------
    # Encoding
    # -------------------------

    sample_sentence = "I love AI"

    sample_tokens = tokenizer.tokenize(sample_sentence)

    encoded = vocab.encode(sample_tokens)

    print("\n" + "=" * 50)
    print("Encoded Sentence")
    print("=" * 50)

    print(sample_tokens)
    print(encoded)

    # -------------------------
    # Decoding
    # -------------------------

    decoded = vocab.decode(encoded)

    print("\n" + "=" * 50)
    print("Decoded Sentence")
    print("=" * 50)

    print(decoded)


if __name__ == "__main__":
    main()