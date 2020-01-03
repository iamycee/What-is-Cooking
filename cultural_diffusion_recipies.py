import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import json
from scipy import sparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

with open('train.json') as data_file:
    df = json.load(data_file)

#print(df.head(10))


# countsMatrix is a matrix where the [i, j] cell is the number of times ingredient j appears in cuisine i
def create_dict_cuisine_ingredients(json):
    dict_cuisine_ingredients = {}
    cuisines = []
    ingredients = []

    for i in range(len(json)):
        cuisine = json[i]['cuisine']

        ingredients_per_cuisine = json[i]['ingredients']

        if cuisine not in dict_cuisine_ingredients.keys(
        ):  #if cuisine not a key yet
            cuisines.append(cuisine)  #append to list
            dict_cuisine_ingredients[
                cuisine] = ingredients_per_cuisine  # add the key-value

        else:  #else if it is a key
            current_list = dict_cuisine_ingredients[
                cuisine]  #current list of ingredients
            current_list.extend(
                ingredients_per_cuisine)  # add more ingredients
            dict_cuisine_ingredients[cuisine] = current_list

            #one ingredient appearing more than once is okay because we need to TF_IDF this baby

        ingredients.extend(ingredients_per_cuisine)

    ingredients = list(set(ingredients))  #unique
    num_unique_ingredients = len(ingredients)
    num_cuisines = len(cuisines)

    return dict_cuisine_ingredients, num_cuisines, num_unique_ingredients, cuisines, ingredients


# Now  we need to prepare the term count matrix were value at [i,j]  is the number of times ingredient j has appeared in cuisine i


def create_term_count_matrix(dictionary, num_cuisines, num_ingredients,
                             cuisines, ingredients):
    term_count_matrix = np.zeros((num_cuisines, num_ingredients))
    i = 0

    for cuisine in cuisines:
        ingredients_per_cuisine = dict_cuisine_ingredients[cuisine]

        for ingredient in ingredients_per_cuisine:
            j = ingredients.index(ingredient)

            term_count_matrix[i, j] += 1

        i += 1

    return term_count_matrix


dict_cuisine_ingredients, num_cuisines, num_ingredients, cuisines, ingredients = create_dict_cuisine_ingredients(
    df)
term_count_matrix = create_term_count_matrix(dict_cuisine_ingredients,
                                             num_cuisines, num_ingredients,
                                             cuisines, ingredients)

# Helpers are done. Now time for TF-IDF

# create a tf-idf matrix of the term_count_matrix, the difference is that the [i,j] cell contains the
# tf-idf weight of ingredient j instead of the counts


def tf_idf_from_count_matrix(term_count_matrix):
    term_count_matrix = sparse.csr_matrix(
        term_count_matrix)  #compressed sparse representation
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(term_count_matrix)
    tfidf.toarray()

    return tfidf


tf_idf_matrix = tf_idf_from_count_matrix(term_count_matrix)

# reduce to 2 dimensions
clf = TruncatedSVD(n_components=2)
reduced_data = clf.fit_transform(tf_idf_matrix)

# convert to dataframe
pca2df = pd.DataFrame(reduced_data)
pca2df.columns = ['PC1', 'PC2']  #column labels

#KMEANS
