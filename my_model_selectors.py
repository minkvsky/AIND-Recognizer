import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
from sklearn.model_selection import GridSearchCV # new added


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        try:
            bestscore = 0
            best_num_components = self.min_n_components
            ls_res = []
            for n_components in [self.min_n_components, self.max_n_components]:
                score = self.bic_model(n_components)[1]
                ls_res.append((n_components, score))
            return max(ls_res, key = lambda x:x[1])[0]
        except:
            return self.base_model(self.n_constant)

    def bic_model(self, n_components):

        model = self.base_model(n)
        logL = model.score(self.X, self.lengths)
        logN = np.log(len(self.X))
        p = n_components ** 2 + 2 * model.n_features * n_components - 1
        bic_score = -2 * logL + p * logN

        return model, bic_score


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        try:
            bestscore = 0
            best_num_components = self.min_n_components
            ls_res = []
            for n_components in [self.min_n_components, self.max_n_components]:
                score = self.dic_model(n_components)[1]
                ls_res.append((n_components, score))
            return max(ls_res, key = lambda x:x[1])[0]
        except:
            return self.base_model(self.n_constant)


    def dic_model(self, n_components):

        model = self.base_model(n)
        scores = []
        for word, (X, lengths) in self.hwords.items():
            if word != self.this_word:
                scores.append(model.score(X, lengths))
        dic_score = model.score(self.X, self.lengths) - np.mean(scores)
        return model, dic_score



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # GridSearchCV failed
        # parameters = {'n_components': [self.min_n_components, self.max_n_components]}
        # hmm_model = GaussianHMM(covariance_type="diag", n_iter=1000,
        #                         random_state=self.random_state, verbose=False)
        # clf = GridSearchCV(hmm_model, parameters, cv=2)
        #
        # # clf.fit(self.sequences, self.lengths)
        # clf.fit(self.X, self.lengths)
        # return clf.best_params_['n_components']

        try:
            bestscore = 0
            best_num_components = self.min_n_components
            ls_res = []
            for n_components in [self.min_n_components, self.max_n_components]:
                score = self.CV_model(n_components)[1]
                ls_res.append((n_components, score))
            return max(ls_res, key = lambda x:x[1])[0]
        except:
            return self.base_model(self.n_constant)

    def CV_model(self, n_components):

        split_method = KFold(2)
        ls_model_score = []
        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            self.X, self.lengths = combine_sequences(cv_train_idx, cv_test_idx)
            model = self.base_model(n_components)
            score = model.score(self.X, self.lengths)
            ls_model_score.append((model, score))

        return max(ls_model_score, key = lambda x:x[1])
