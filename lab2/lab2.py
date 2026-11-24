import gensim

#В каталоге есть файл текст.txt, котором я подбором занимался и логики до которой я дошел
'''
Комбинации:
POSITIVE = ["кокос", "пылинка"]
NEGATIVE = ["скорлупка"]

Лучший вариант ниже получился:
POSITIVE = ["кокос", "пылинка"]
NEGATIVE = ["шерстинка"]

POSITIVE = ["кокос", "пылинка"]
NEGATIVE = ["шерстинка"]

POSITIVE = ["кокос", "пылинка"]
NEGATIVE = ["пульверизатор"]
'''

MODEL_FILE = "cbow.txt"
POSITIVE = ["кокос", "пылинка"]
NEGATIVE = ["пульверизатор"]
TARGET_WORDS = ["финик", "песчинка"]
TOP_N = 10


def find_similar_nouns(model):
    positive = [word + "_NOUN" for word in POSITIVE]
    negative = [word + "_NOUN" for word in NEGATIVE]

    for word in positive + negative:
        if word not in model:
            raise ValueError(f"Слова {word} нет в модели")

    # Складываем векторы positive и вычитаем векторы negative.
    similar_words = model.most_similar(
        positive=positive,
        negative=negative,
        topn=len(model)
    )

    result = []
    for word, similarity in similar_words:
        if word.endswith("_NOUN"):
            result.append((word[:-5], similarity))
            if len(result) == TOP_N:
                break

    return result


def main():
    model = gensim.models.KeyedVectors.load_word2vec_format(
        MODEL_FILE, binary=False
    )

    result = find_similar_nouns(model)
    expression = " + ".join(POSITIVE)
    if NEGATIVE:
        expression += " - " + " - ".join(NEGATIVE)
    print(f"\nКомбинация: {expression}\n")

    for number, (word, similarity) in enumerate(result, start=1):
        print(f"{number:2}. {word:15} {similarity:.4f}")

    print("\nПроверка заданных слов:")
    found_words = [word for word, similarity in result]
    for word in TARGET_WORDS:
        if word in found_words:
            print(f"{word}: место {found_words.index(word) + 1}")
        else:
            print(f"{word}: не входит в первую десятку")


if __name__ == "__main__":
    main()