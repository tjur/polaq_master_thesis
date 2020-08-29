import matplotlib.pyplot as plt


def plot_squad_answers_translations_count():
    x = [1, 2, 3, 4, 5]
    y = [23517, 15513, 2353, 553, 18]

    plt.bar(x, height=y, color="blue", alpha=0.5)
    plt.xticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5+"])
    plt.xlabel("Liczba tłumaczeń")
    plt.ylabel("Liczba odpowiedzi (bez OTHER)")
    plt.show()


def plot_squad_answers_translation_words_count():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [57070, 51935, 27951, 14237, 8879, 5255, 3349, 1839, 903, 127]

    plt.bar(x, height=y, color="green", alpha=0.5)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"])
    plt.xlabel("Liczba słów w odpowiedzi")
    plt.ylabel("Liczba odpowiedzi")
    plt.show()


plot_squad_answers_translations_count()
plot_squad_answers_translation_words_count()
