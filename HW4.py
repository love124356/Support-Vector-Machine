import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# np.set_printoptions(threshold=10)

# ## Load data
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

# 550 data with 300 features
print(x_train.shape)

# It's a binary classification problem
print(np.unique(y_train))


# ## Question 1
# K-fold data partition: Implement the K-fold cross-validation function.
# Your function should take K as an argument
# and return a list of lists (len(list) should equal to K),
# which contains K elements. Each element is a list contains two parts,
# the first part contains the index of all training folds,
# e.g. Fold 2 to Fold 5 in split 1.
# The second part contains the index of validation fold, e.g. Fold 1 in split 1

# Ref: sklearn.model_selection _split.py

def get_test_mask(n_samples, indices, start, stop):
    test_index = indices[start:stop]
    test_mask = np.zeros(n_samples, dtype=bool)
    test_mask[test_index] = True
    return test_mask


def cross_validation(X_train, y_train, shuffle=True, k=5, random_state=35):
    n_samples = X_train.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        rstate = np.random.RandomState(random_state)
        rstate.shuffle(indices)
    n_splits = k
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    KFold = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_mask = get_test_mask(n_samples, indices, start, stop)
        train_index = indices[np.logical_not(test_mask)]
        test_index = indices[test_mask]
        # error = [x for x in train_index if x in test_index]
        # print(len(error))
        KFold.append([train_index, test_index])
        current = stop
    return KFold


kfold_data = cross_validation(x_train, y_train, k=10)
# for i, (train_index, val_index) in enumerate(kfold_data):
#     print(
#         "Split: %s, Training index: %s, Validation index: %s" %
#         (i+1, train_index, val_index)
#     )

# should contain 10 fold of data
assert len(kfold_data) == 10
# each element should contain train fold and validation fold
assert len(kfold_data[0]) == 2
# The number of data in each validation fold
# should equal to training data divieded by K
assert kfold_data[0][1].shape[0] == 55

# For example
X = np.arange(20)
kf = cross_validation(X, y_train, k=10)
print('Test index of cross validation')
print('-' * 10)
for i, (train_index, val_index) in enumerate(kf):
    print(
        "Split: %s, Training index: %s, Validation index: %s" %
        (i+1, train_index, val_index)
    )
print('-' * 10)


# ## Question 2
# Using sklearn.svm.SVC to train a classifier on the provided train set and
# conduct the grid search of “C”, “kernel” and “gamma”
# to find the best parameters by cross-validation.

def gridsearch(x, y, kfold_data, candidate_C, candidate_gamma):
    history = []
    max_acc = 0
    n_gamma = len(candidate_gamma)
    candidate = [(c, g) for c in candidate_C for g in candidate_gamma]
    tmp_acc = []
    for i, (c, g) in enumerate(candidate):
        avg_acc = 0
        for j, (train, test) in enumerate(kfold_data):
            clf = SVC(C=c, kernel='rbf', gamma=g)
            clf.fit(x[train], y[train])
            y_pred = clf.predict(x[test])
            acc = accuracy_score(y[test], y_pred)
            avg_acc += acc
        avg_acc /= len(kfold_data)
        # print(f'C={c}, gamma={g}, average accuracy={avg_acc:.3f}')
        tmp_acc.append(avg_acc)

        if avg_acc > max_acc:
            best_C = c
            best_gamma = g
            max_acc = avg_acc

        if i % n_gamma == n_gamma - 1:
            history.append(tmp_acc)
            tmp_acc = []

    return np.asarray(history), (best_C, best_gamma)


candidate_C = [1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4]
candidate_gamma = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3]
history, best_parameters = gridsearch(
    x_train, y_train, kfold_data, candidate_C, candidate_gamma
)
print(f'Best parameter (C, gamma): {best_parameters}')

# ## Question 3
# Plot the grid search results of your SVM.
# The x, y represents the hyperparameters of “gamma” and “C”, respectively.
# And the color represents the average score of validation folds
# You reults should be look like the reference image
# https://miro.medium.com/max/1296/1*wGWTup9r4cVytB5MOnsjdQ.png

# Ref: https://reurl.cc/3o4Lk8, https://reurl.cc/j1y56n


def plot_grid_search(history):
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(history, interpolation='nearest', cmap=plt.cm.coolwarm)
    for y in range(history.shape[0]):
        for x in range(history.shape[1]):
            plt.text(
                x, y, '%.2f' % history[y][x],
                horizontalalignment='center',
                verticalalignment='center',
                color='white'
            )
    plt.xlabel('Gamma Parameter')
    plt.ylabel('C Parameter')
    plt.colorbar()
    plt.xticks(np.arange(len(candidate_gamma)), candidate_gamma)
    plt.yticks(np.arange(len(candidate_C)), candidate_C)
    plt.title('Hyperparameter Gridsearch')
    plt.savefig('Gridsearch.png', dpi=300)
    # plt.show()


plot_grid_search(history)

# ## Question 4
# Train your SVM model by the best parameters you found
# from question 2 on the whole training set and
# evaluate the performance on the test set.

best_C, best_gamma = best_parameters
best_model = SVC(C=best_C, kernel='rbf', gamma=best_gamma)
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
print("Accuracy score: ", accuracy_score(y_pred, y_test))
