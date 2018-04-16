import sys
from utils import *
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('TkAgg')
from pylab import *


def LDA_model(input, n_components, n_iter):
    norm_corpus = normalize_corpus(input)
    vectorizer, _ = build_feature_matrix(norm_corpus, feature_type='tfidf')

    lda_model = LatentDirichletAllocation(n_components=n_components,  # Number of topics
                                          max_iter=n_iter,  # Max learning iterations
                                          learning_method='online',
                                          random_state=100,  # Random state
                                          batch_size=128,  # n docs in each learning iter
                                          evaluate_every=-1,
                                          n_jobs=-1)

    svd_transformer = Pipeline([('tfidf', vectorizer),
                                ('svd', lda_model)])

    svd_matrix = svd_transformer.fit_transform(norm_corpus)

    return vectorizer, lda_model, svd_transformer, svd_matrix


if __name__ == '__main__':
    input_data = ["The fox jumps over the dog",
                  "The fox is very clever and quick",
                  "The dog is slow and lazy",
                  "The cat is smarter than the fox and the dog",
                  "Python is an excellent programming language",
                  "Java and Ruby are other programming languages",
                  "Python and Java are very popular programming languages",
                  "Python programs are smaller than Java programs"]

    vectorizer, svd_model, svd_transformer, svd_matrix = LDA_model(input_data, 2, 100)

    # test_query = ['dog fox jump smarter cat']
    test_query = ['program small java python', 'dog fox jump smarter cat']

    norm_query = normalize_corpus(test_query)
    query_vector = svd_transformer.transform(norm_query)
    predicted_topic = np.argmax(query_vector, axis=1).tolist()

    # print query_vector
    print 'test query predicted topic: ', predicted_topic

    feat_names = vectorizer.get_feature_names()

    for compNum in range(len(svd_model.components_)):
        print compNum
        comp = svd_model.components_[compNum]

        # Sort the weights in the first component, and get the indices
        indices = np.argsort(comp).tolist()[::-1]

        # Grab the top 10 terms which have the highest weight in this component.
        terms = [feat_names[weightIndex] for weightIndex in indices[0:10]]
        weights = [comp[weightIndex] for weightIndex in indices[0:10]]
        # print terms, weights
        terms.reverse()
        weights.reverse()
        print terms, weights

        positions = arange(10) + .5  # the bar centers on the y axis

        figure(compNum)
        barh(positions, weights, align='center')
        yticks(positions, terms)
        xlabel('Weight')
        title('Strongest terms for component %d' % (compNum))
        grid(True)
        show()
