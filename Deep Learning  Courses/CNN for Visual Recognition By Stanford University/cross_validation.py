import numpy as np
import matplotlib.pyplot as plt 
from assignment1.cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor




def cross_validation(X_train, y_train, num_folds=5):
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = []
    y_train_folds = []
    ################################################################################
    # TODO:                                                                        #
    # Split up the training data into folds. After splitting, X_train_folds and    #
    # y_train_folds should each be lists of length num_folds, where                #
    # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
    # Hint: Look up the numpy array_split function.                                #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    X_train_folds = np.array_split(X_train, num_folds) 
    y_train_folds = np.array_split(y_train, num_folds) 

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # A dictionary holding the accuracies for different values of k that we find
    # when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the different
    # accuracy values that we found when using that value of k.
    k_to_accuracies = {}


    ################################################################################
    # TODO:                                                                        #
    # Perform k-fold cross validation to find the best value of k. For each        #
    # possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
    # where in each case you use all but one of the folds as training data and the #
    # last fold as a validation set. Store the accuracies for all fold and all     #
    # values of k in the k_to_accuracies dictionary.                               #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    for k in k_choices:
        accuracies_temp = []
        for fold in range(num_folds):
            #Folding
            X_train_temp = X_train_folds[fold]
            y_train_temp = y_train_folds[fold]
            X_test_temp = np.concatenate(np.delete(X_train_folds,fold,0), axis= 0)
            y_test_temp = np.concatenate(np.delete(y_train_folds,fold,0), axis= None)

            #Trainig
            classifier = KNearestNeighbor()
            classifier.train(X_train_temp, y_train_temp)
            dists = classifier.compute_distances_no_loops(X_test_temp)

            #Evaluation
            num_test_temp = X_test_temp.shape[0]
            y_test_pred_temp= classifier.predict_labels(dists, k=k)
            num_correct_temp = np.sum(y_test_pred_temp == y_test_temp)
            accuracy_temp = float(num_correct_temp) / num_test_temp
            accuracies_temp.append(accuracy_temp)
        k_to_accuracies[k] = accuracies_temp
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
    #Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))


    # plot the raw observations
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v)
                                for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v)
                               for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()
    return k_to_accuracies
