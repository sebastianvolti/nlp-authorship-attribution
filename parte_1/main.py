import zipf
import pandas as pd


def main():
    # Parte 1
    authors = ['L1-Conrad', 'L2-Zola', 'L3-Proust', 'L4-Austen', 'L5-Flaubert']
    frecuencies = {}

    for i, author in enumerate(authors):
        tokenized_text = zipf.tokenize_text(f'textos/ConAutor/{author}.txt')
        frecuency_values = sorted([x for x in zipf.get_frecuencies(tokenized_text).values()], reverse=True)
        frecuencies[author] = frecuency_values

    print([fv[:10] for fv in frecuencies.values()])

    # Parte 2
    plots = []
    for i, text in enumerate(authors):
        plots.append((list(range(len(frecuency_values))), frecuency_values, authors[i]))

    # zipf.plot(plots)

    # Parte 3
    absolute_frecuencies = []
    relative_frecuencies = []

    for author_frecuencies in frecuencies.values():
        count = len(list(filter(lambda x: x == 1, author_frecuencies)))
        absolute_frecuencies.append(count)
        relative_frecuencies.append("{0:.2f}".format(count/len(author_frecuencies)))

    table = {
        'absolute_frecuencies': absolute_frecuencies,
        'relative_frecuencies': relative_frecuencies
    }
    dfObj = pd.DataFrame.from_dict(table, orient='index', columns=authors) 
    print(dfObj)


if __name__ == '__main__':
    main()
