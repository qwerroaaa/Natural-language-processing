import re
import pymorphy3
from nltk.tokenize import sent_tokenize, word_tokenize

INPUT_FILE = "input.txt"
TARGET_POS = {"NOUN", "ADJF"}
WORD_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)*", re.IGNORECASE)
morph = pymorphy3.MorphAnalyzer()

# Какие-то морфлогические причуды, которые лучше упростить
# gen2 - тоже родительный, но из разряда "чаю"
# loc2 - предложный "в лесу" 
# acc2 - винительный
# чисто технически такое решение дает "в темном лесу" добро, на то чтобы вывестись в консоль
CASE_NORMALIZATION = {"gen2": "gent", "loc2": "loct", "acc2": "accs"}

morph = pymorphy3.MorphAnalyzer()


def get_gender(tag):
    # Обработка общего рода, т.е. слова а-ля Сирота, Жадина и т.д.
    if "ms-f" in tag:
        return "ms-f"
    return str(tag.gender) if tag.gender else None


def genders_agree(g1, g2):
    if g1 is None or g2 is None:
        return False
    if g1 == g2:
        return True
    return (
        (g1 == "ms-f" and g2 in {"masc", "femn"})
        or (g2 == "ms-f" and g1 in {"masc", "femn"})
    )


def tags_agree(t1, t2):
    # Сравниваем признаки исходных форм, до лемматизации
    # Согласованность по числу
    if t1.number is None or t1.number != t2.number:
        return False

    # Согласованность по падежу
    case1 = CASE_NORMALIZATION.get(t1.case, t1.case)
    case2 = CASE_NORMALIZATION.get(t2.case, t2.case)
    if case1 is None or case1 != case2:
        return False

    # Для множественного числа
    if t1.number == "plur" and (t1.POS == "ADJF" or t2.POS == "ADJF"):
        return True

    gender1, gender2 = get_gender(t1), get_gender(t2)

    # проверка согласованности по роду
    return genders_agree(gender1, gender2)


def candidate_parses(token):
    if not WORD_RE.fullmatch(token):
        return []

    parses = morph.parse(token)
    top_pos = parses[0].tag.POS
    if top_pos not in TARGET_POS:
        return []
    return [p for p in parses if p.tag.POS == top_pos]


def match_pair(parses1, parses2):
    # Проверяем разборы токенов на согласованность
    best_pair = None
    best_score = -1.0

    for p1 in parses1:
        for p2 in parses2:
            if tags_agree(p1.tag, p2.tag):
                score = p1.score * p2.score
                if score > best_score:
                    best_pair = (p1.normal_form, p2.normal_form)
                    best_score = score

    return best_pair


def find_pairs(text):
    pairs = []

    for sentence in sent_tokenize(text, language="russian"):
        tokens = word_tokenize(
            sentence, language="russian", preserve_line=True
        )

        analyses = [candidate_parses(token) for token in tokens]

        for parses1, parses2 in zip(analyses, analyses[1:]):
            pair = match_pair(parses1, parses2)
            if pair is not None:
                pairs.append(pair)

    return pairs


def main():
    with open(INPUT_FILE, encoding="utf-8-sig") as f:
        pairs = find_pairs(f.read())

    for word1, word2 in pairs:
        print(word1, word2)


if __name__ == "__main__":
    main()
