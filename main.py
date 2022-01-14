import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.decomposition import PCA
#from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix, precision_score, recall_score, average_precision_score, matthews_corrcoef, roc_auc_score, confusion_matrix, plot_roc_curve
import matplotlib.pyplot as plt
from matplotlib.pyplot import matshow
import time
from sklearn.naive_bayes import GaussianNB
import os

def validateInput(user_input, param_list):
    algorithm_list = param_list
    algorithm_list.extend(['quit', 'review'])
    while (user_input not in algorithm_list):
        user_input = str(input("Enter a valid algorithm name, 'review', or 'quit': \ninput> "))
    algorithm_list.remove('quit')
    algorithm_list.remove('review')
    return user_input


def dataToFile(data, path):
    print("\n---------------------------------------------------------------\n")
    print("\nWriting evaluation metrics to file...")
    f = open(path, 'w')
    for item in range(len(data)):
        f.writelines(f"{data[item]}\n")
    f.close()
    print("\nFinished saving data!")
    print("\n---------------------------------------------------------------\n")
    return


def performance_review(Y_test, Y_pred, start, end, Y_hat_train, Y_train, clf):
    #average precision-recall score
    #y_score = clf.decision_function(X_test)
    #print(f"Average precision-recall score: {round(average_precision_score(Y_test, y_score), 2)}")

    #precision score
    p = precision_score(Y_test, Y_pred)
    print(f"Precision score: {round(p*100, 2)}")

    #recall score
    r = recall_score(Y_test, Y_pred)
    print(f"Recall score: {round(r*100, 2)}")

    #Matthews correlation coefficient
    mcc = matthews_corrcoef(Y_test, Y_pred)
    print(f"Matthew's Correlated Coefficient: {round(mcc*100, 2)}")

    #Area Under the Receiver Operating Characteristic Curve scores
    auc_and_roc = roc_auc_score(Y_test, Y_pred)
    print(f"Area Under the Receiver Operating Characteristic Curve scores: {round(auc_and_roc*100, 2)}")
    
    #Confusion Matrix
    confusion = confusion_matrix(Y_test, Y_pred)
    print(f"Confusion matrix: {confusion}")

    #F1 Score
    f1 = f1_score(Y_test, Y_pred)
    print(f"F1 Score: {round(f1, 2)}")

    #Training time
    time_to_complete = end-start
    print(f'Time to train: {round(time_to_complete, 2)} seconds')

    #No. Missclassified Samples
    total_misclassified = (Y_test != Y_pred).sum()
    print(f'Misclassified samples: {total_misclassified}')

    #Accuracy Score
    acc = accuracy_score(Y_test, Y_pred)
    print(f'Accuracy: {round(acc*100, 2)}%\n')
    
    return p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc


def visualizeConfusionMatrix(clf, X_test, Y_test):
    disp = plot_confusion_matrix(clf, X_test, Y_test, cmap='Blues', normalize='true', display_labels=['without', 'with'])
    disp.ax_.set_title("Confusion Matrix")
    print("CONFUSION MATRIX")
    print(disp.confusion_matrix)
    plt.show()
    return


def visualizeROC(name, clf, X_test, Y_test):
    ax = plt.gca()

    if name == "logistic":
        global log_disp
        log_disp = plot_roc_curve(clf, X_test, Y_test, ax=ax, alpha=1)
    elif name == "perceptron":
        global per_disp
        per_disp = plot_roc_curve(clf, X_test, Y_test, ax=ax, alpha=0.9)
    elif name == "rf":
        global rf_disp
        rf_disp = plot_roc_curve(clf, X_test, Y_test, ax=ax, alpha=0.8)
    elif name == "dt":
        global dt_disp
        dt_disp = plot_roc_curve(clf, X_test, Y_test, ax=ax, alpha=0.7)
    elif name == "linear":
        global linear_disp
        linear_disp = plot_roc_curve(clf, X_test, Y_test, ax=ax, alpha=0.6)
    elif name == "knn":
        global knn_disp
        knn_disp = plot_roc_curve(clf, X_test, Y_test, ax=ax, alpha=0.5)
    elif name == "svm":
        global svm_disp
        svm_disp = plot_roc_curve(clf, X_test, Y_test, ax=ax, alpha=0.4)
    elif name == "naive":
        global naive_disp
        naive_disp = plot_roc_curve(clf, X_test, Y_test, ax=ax, alpha=0.3)
    
    disp_plot_list = ['log_disp', 'per_disp', 'rf_disp', 'dt_disp', 'linear_disp', 'knn_disp', 'svm_disp', 'naive_disp']

    for item in disp_plot_list:
        if item == 'log_disp':
            if item in globals():
                log_disp.plot(ax=ax, alpha=1)
        elif item == 'per_disp':
            if item in globals():
                per_disp.plot(ax=ax, alpha=0.9)
        elif item == 'rf_disp':
            if item in globals():
                rf_disp.plot(ax=ax, alpha=0.8)
        elif item == 'dt_disp':
            if item in globals():
                dt_disp.plot(ax=ax, alpha=0.7)
        elif item == 'linear_disp':
            if item in globals():
                linear_disp.plot(ax=ax, alpha=0.6)
        elif item == 'knn_disp':
            if item in globals():
                knn_disp.plot(ax=ax, alpha=0.5)
        elif item == 'svm_disp':
            if item in globals():
                svm_disp.plot(ax=ax, alpha=0.4)
        elif item == 'naive_disp':
            if item in globals():
                naive_disp.plot(ax=ax, alpha=0.3)
    
    plt.show()
    return

def logistic_regression(X_train, X_test, Y_train, Y_test):
    print("\n---------- Logistic Regression ----------")
    start = time.time()

    clf = LogisticRegression(max_iter=250000)
    clf.fit(X_train, Y_train.values.ravel())
    Y_pred = clf.predict(X_test)
    Y_pred = Y_pred.reshape(149993, 1)
    Y_hat_train = clf.predict(X_train)

    end = time.time()
    p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc = performance_review(Y_test, Y_pred, start, end, Y_hat_train, Y_train, clf)
    data = [p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc]
    
    for item in range(len(data)):
        data[item] = str(data[item])

    path = r"performance_results/logistic_regression"
    dataToFile(data, path)
    visualizeConfusionMatrix(clf, X_test, Y_test)
    visualizeROC("logistic", clf, X_test, Y_test)
    return


def perceptron(X_train, X_test, Y_train, Y_test):
    print("\n---------- Perceptrons ----------")
    start = time.time()

    clf = MLPClassifier(max_iter=250000, random_state=0)
    clf.fit(X_train, Y_train.values.ravel())
    Y_pred = clf.predict(X_test)
    Y_pred = Y_pred.reshape(149993, 1)
    Y_hat_train = clf.predict(X_train)

    end = time.time()
    p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc = performance_review(Y_test, Y_pred, start, end, Y_hat_train, Y_train, clf)
    data = [p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc]
    path = r"performance_results/perceptron"
    dataToFile(data, path)
    visualizeConfusionMatrix(clf, X_test, Y_test)
    visualizeROC("perceptron", clf, X_test, Y_test)
    return


def support_vector_machine(X_train, X_test, Y_train, Y_test):
    print("\n---------- Support Vector Machines ----------")
    
    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    kernel_list.sort()
    kernel_choice = ''

    while kernel_choice not in kernel_list:
        print("Enter the name of the kernel you would like to use: \n")
        for kernel in kernel_list:
            print(kernel)
        kernel_choice = str(input("\ninput> "))
        kernel_choice = validateInput(kernel_choice, kernel_list)

    start = time.time()
    svm = SVC(kernel=kernel_choice, C=1.0, random_state=1, verbose=1)
    svm.fit(X_train, Y_train.values.ravel())
    Y_pred = svm.predict(X_test)
    Y_pred = Y_pred.reshape(149993, 1)
    Y_hat_train = svm.predict(X_train)

    end = time.time()
    p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc = performance_review(Y_test, Y_pred, start, end, Y_hat_train, Y_train, svm)
    data = [p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc]
    for item in data:
        item = str(item)
    path = r"performance_results/svm"
    dataToFile(data, path)
    visualizeConfusionMatrix(svm, X_test, Y_test)
    visualizeROC("svm", svm, X_test, Y_test)
    return


def random_forests(X_train, X_test, Y_train, Y_test):
    print("\n---------- Random Forests ----------")
    start = time.time()

    clf = RandomForestClassifier(n_estimators = 100)
    clf.fit(X_train, Y_train.values.ravel())
    Y_pred = clf.predict(X_test)
    Y_pred = Y_pred.reshape(149993, 1)
    Y_hat_train = clf.predict(X_train)

    end = time.time()
    p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc = performance_review(Y_test, Y_pred, start, end, Y_hat_train, Y_train, clf)
    data = [p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc]
    path = r"performance_results/random_forest"
    dataToFile(data, path)
    visualizeConfusionMatrix(clf, X_test, Y_test)
    visualizeROC("rf",clf, X_test, Y_test)
    return


def decision_tree(X_train, X_test, Y_train, Y_test):
    print("\n---------- Decision Trees ----------")
    start = time.time()

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, Y_train.values.ravel())
    Y_pred = clf.predict(X_test)
    Y_pred = Y_pred.reshape(149993, 1)
    Y_hat_train = clf.predict(X_train)
    
    end = time.time()
    p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc = performance_review(Y_test, Y_pred, start, end, Y_hat_train, Y_train, clf)
    data = [p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc]
    path = r"performance_results/decision_tree"
    dataToFile(data, path)
    visualizeConfusionMatrix(clf, X_test, Y_test)
    visualizeROC("dt",clf, X_test, Y_test)
    return


def knn(X_train, X_test, Y_train, Y_test):
    def config_knn(w='uniform', a='auto'):
        start = time.time()

        clf = KNeighborsClassifier(weights=w, algorithm=a)
        clf.fit(X_train, Y_train.values.ravel())
        Y_pred = clf.predict(X_test)
        Y_pred = Y_pred.reshape(149993, 1)
        Y_hat_train = clf.predict(X_train)

        end = time.time()
        p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc = performance_review(Y_test, Y_pred, start, end, Y_hat_train, Y_train, clf)
        data = [p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc]
        path = r"performance_results/knn"
        dataToFile(data, path)
        visualizeConfusionMatrix(clf, X_test, Y_test)
        visualizeROC("knn", clf, X_test, Y_test)
        return

    print("\n---------- K Nearest Neighbors ----------")

    #Getting user input
    default_or_config = str(input("Enter 'default' to use default settings, or 'config' to customise the classifier:\n "))
    
    if default_or_config == 'config':
        weights = ['uniform', 'distance']
        algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        
        weights_choice = ""
        algorithm_choice =""

        while (weights_choice not in weights) and (algorithm_choice not in algorithm):
            weights_choice = str(input(f"Enter the weight function used in prediction ({weights}): \n"))
            algorithm_choice = str(input(f"Enter the algorithm to be used for computing nearest neighbours ({algorithm}): \n"))
        
        print("The Minkowski distance metric is in use\n ")

        if (algorithm_choice == 'ball_tree') or (algorithm_choice == 'kd_tree'):
            leaf_size_input = int(input("Enter the leaf size (deault is 30)\n"))
            start = time.time()
            clf = KNeighborsClassifier(weights=weights_choice, algorithm=algorithm_choice, leaf_size=leaf_size_input)
            clf.fit(X_train, Y_train.values.ravel())
            Y_pred = clf.predict(X_test)
            Y_pred = Y_pred.reshape(149993, 1)
            Y_hat_train = clf.predict(X_train)
            end = time.time()
            p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc = performance_review(Y_test, Y_pred, start, end, Y_hat_train, Y_train, clf)
            data = [p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc]
            path = r"performance_results/knn"
            dataToFile(data, path)
            visualizeConfusionMatrix(clf, X_test, Y_test)
            visualizeROC("knn", clf, X_test, Y_test)

        else:
            config_knn(weights_choice, algorithm_choice)
        
    elif default_or_config == 'default':
        config_knn()
    
    else:
        print("You did not enter a valid response, returning to main menu... \n")

    return


def pca(X_train, X_test, Y_train, Y_test):
    print("\n---------- Principal Component Analysis ----------")
    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_pca = pca.transform(X_train)
    print(f"Original shape: {X_train.shape}")
    print(f"Reduced shape: {X_pca.shape}\n")
    print(f"PCA component shape: {pca.components_.shape}")
    plt.matshow(pca.components_, cmap = 'viridis')
    plt.yticks([0, 1], ["First Component", "Second Component"])
    plt.colorbar()
    plt.xticks(range(len(df.columns)), df.columns.values, rotation=60, ha="left")
    plt.xlabel("Feature")
    plt.ylabel("Principal Components")
    return

'''
##PYTHON COULD NOT LOCATE THE SELFTRAININGCLASSIFIER MODULE, SO IT HAS BEEN LEFT OUT OF THE FINAL BUILD
def semi_supervised(X_train, X_test, Y_train, Y_test):
    print("Now Using Semi-Supervised Classification: \n")
    print("Logistic Regression...\n")
    clf = LogisticRegression()
    self_training_model = SelfTrainingClassifier(clf)
    self_training_model.fit(X_train, Y_train.values.ravel())
    y_hat_test = clf.predict(X_test)
    y_hat_train = clf.predict(X_train)
    train_f1 = f1_score(Y_train, y_hat_train)
    test_f1 = f1_score(Y_test, y_hat_test)
    print(f"Train f1 Score: {train_f1}")
    print(f"Test f1 Score: {test_f1}")
    plot_confusion_matrix(clf, X_test, Y_test, cmap='Blues', normalize='true', display_labels=['without', 'with'])
    clf.predict_proba(X_test)
    return
'''

def linear_svc(X_train, X_test, Y_train, Y_test):
    print("\n---------- Linear Support Vector Machine ----------")
    
    loss_list = ['hinge', 'squared_hinge']
    loss_choice = ''

    while loss_choice not in loss_list:
        print("Enter the lose function you would like to use: \n")
        for loss in loss_list:
            print(loss)
        loss_choice = str(input("\ninput> "))
        loss_choice = validateInput(loss_choice, loss_list)
    
    start = time.time()
    clf = LinearSVC(verbose=1, max_iter=100, penalty='l2', loss=loss_choice)
    clf.fit(X_train, Y_train.values.ravel())
    Y_pred = clf.predict(X_test)
    Y_pred = Y_pred.reshape(149993, 1)
    Y_hat_train = clf.predict(X_train)
    end = time.time()
    
    p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc = performance_review(Y_test, Y_pred, start, end, Y_hat_train, Y_train, clf)
    data = [p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc]
    path = r"performance_results/linear_svc"
    dataToFile(data, path)
    visualizeConfusionMatrix(clf, X_test, Y_test)
    visualizeROC("linear", clf, X_test, Y_test)
    return


def bayes(X_train, X_test, Y_train, Y_test):
    print("\n---------- Gaussian Naive Bayes ----------")

    start = time.time()
    clf = GaussianNB()
    clf.fit(X_train, Y_train.values.ravel())
    Y_pred = clf.predict(X_test)
    Y_pred = Y_pred.reshape(149993, 1)
    Y_hat_train = clf.predict(X_train)
    end = time.time()

    p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc = performance_review(Y_test, Y_pred, start, end, Y_hat_train, Y_train, clf)
    data = [p, r, mcc, auc_and_roc, confusion, f1, time_to_complete, total_misclassified, acc]
    path = r"performance_results/naive_bayes"
    dataToFile(data, path)
    visualizeConfusionMatrix(clf, X_test, Y_test)
    visualizeROC("naive", clf, X_test, Y_test)
    return


def readResults():
    print("\n---------- Performance Review ----------")
    path = r"performance_results"
    files = os.listdir(path)

    print("\nEnter the filename you would like to review the performance of: \n")
    for f in files:
        print(f)
    user_input = str(input("\ninput> "))
    validateInput(user_input, files)
    path = f"{path}/{user_input}"

    chosen_file = open(path, 'r')
    results = chosen_file.readlines()
    print(f"\n---------- {user_input.upper()} RESULTS ----------\n")
    '''
    for line in results:
        print(line)
    '''

    for x in range(len(results)):
        if x == 0:
            print(f"Precision Score: {round(float(results[x])*100, 2)}%\n")
        elif x == 1:
            print(f"Recall Score: {round(float(results[x])*100, 2)}%\n")
        elif x == 2:
            print(f"Matthew's Correlated Coefficient: {round(float(results[x])*100, 2)}%\n")
        elif x == 3:
            print(f"Area Under the Receiver Operating Characteristic Curve scores: {round(float(results[x])*100, 2)}%\n")
        elif x == 6:
            print(f"F1 Score: {round(float(results[x])*100, 2)}%\n")
        elif x == 7:
            print(f"Time to train: {round(float(results[x]), 2)} seconds\n")
        elif x == 8:
            results[x] = results[x].replace('Class', '')
            print(f"Misclassified samples: {int(results[x])}\n")
        elif x ==10:
            print(f"Accuracy: {round(float(results[x])*100, 2)}%\n")
        x+=1
    
    print("-----------------------------------------------------\n")

    return

##Main program
print("\n---------------------------------------------------------------\n")
print("\nThis program is designed to facilitate testing of Supervised Machine Learning algorithms on preprocessed tcpflow data\n")
print("\n---------------------------------------------------------------\n")
print("Loading the dataset from file... \n")

path = r'csv_files/final_preprocessed.csv'

df = pd.read_csv(path, sep=',', low_memory=False)

df = df.drop(['Unnamed: 0', 'index'], axis=1)

X = df.iloc[:, :-1]
Y = df.iloc[:, -1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

print("Applying StandardScaling to the dataset... \n")

sc_x = StandardScaler()

X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

algorithm_list = ['logistic regression', 'perceptrons', 'support vector machines', 'random forests', 'decision trees', 'knn', 'linear svc', 'naive bayes', 'pca']
algorithm_list.sort()

print("\n---------------------------------------------------------------\n")

while True:
    
    print("Enter the name of one of the following algorithms to begin testing, 'review' to review the performance of a classifier, or 'quit' to exit out of the program: \n")

    for algorithm in algorithm_list:
        print(algorithm)
    
    user_input = str(input("\ninput> "))
    user_input = validateInput(user_input, algorithm_list)

    if user_input == 'logistic regression':
        logistic_regression(X_train, X_test, Y_train, Y_test)
    elif user_input == 'perceptrons':
        perceptron(X_train, X_test, Y_train, Y_test)
    elif user_input == 'support vector machines':
        support_vector_machine(X_train, X_test, Y_train, Y_test)
    elif user_input == 'random forests':
        random_forests(X_train, X_test, Y_train, Y_test)
    elif user_input == 'decision trees':
        decision_tree(X_train, X_test, Y_train, Y_test)
    elif user_input == 'pca':
        pca(X_train, X_test, Y_train, Y_test)
    elif user_input == 'knn':
        knn(X_train, X_test, Y_train, Y_test)
    elif user_input == 'linear svc':
        linear_svc(X_train, X_test, Y_train, Y_test)
    elif user_input == 'naive bayes':
        bayes(X_train, X_test, Y_train, Y_test)
    elif user_input == 'review':
        readResults()
    else:
        print("\nYOU ENTERED AN INCORRECT VALUE, OR CHOOSE TO QUIT THE PROGRAM...")
        quit()
