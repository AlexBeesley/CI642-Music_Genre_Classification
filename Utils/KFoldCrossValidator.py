from sklearn.model_selection import KFold


class KFoldCrossValidator:
    def __init__(self, model, X, y, k=5):
        self.model = model
        self.X = X
        self.y = y
        self.k = k

    def validate(self):
        kfold = KFold(n_splits=self.k)
        scores = []
        for train_index, test_index in kfold.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.model.fit(X_train, y_train, epochs=10, verbose=0)
            score = self.model.evaluate(X_test, y_test, verbose=0)
            scores.append(score[1])
        return sum(scores) / len(scores)
