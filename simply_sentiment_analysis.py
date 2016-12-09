import pandas as pd

from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


def find_best_params(data):

    df = pd.read_csv(data, header=0, delimiter='\t')
    data_rows = (len(df))

    print('Training on {0} rows'.format(str(data_rows)))

    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])

    parameters = dict(vect__max_df=(0.25, 0.5, 1), vect__ngram_range=((1, 1), (1, 2)), vect__use_idf=(True, False),
                      vect__stop_words=('english', None), vect__max_features=(5000, 10000, None),
                      vect__norm=('l1', 'l2'), clf__penalty=('l1', 'l2'), clf__C=(0.1, 1, 10))

    X, y = df['Phrase'], df['Sentiment'].as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,
                               verbose=1, scoring='accuracy')

    grid_search.fit(X_train, y_train)

    print('Best score: {0}'.format(grid_search.best_score_))
    print('Best parameters set:')
    best_parameters = grid_search.best_estimator_.get_params()

    best_params = defaultdict(dict)

    for param_name in sorted(parameters.keys()):
        best_params[param_name] = best_parameters[param_name]
        print('\t{0}: {1}'.format(param_name, best_parameters[param_name]))

    predictions = grid_search.predict(X_test)

    print('Accuracy:', accuracy_score(y_test, predictions))
    print('Confusion Matrix:', confusion_matrix(y_test, predictions))
    print('Classification Report:', classification_report(y_test, predictions))


if __name__ == '__main__':

    path_to_training_data = 'train.tsv'
    find_best_params(path_to_training_data)
