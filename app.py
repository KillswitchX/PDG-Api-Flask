import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
import seaborn as sns
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import werkzeug
from IPython.display import Image
from flask import send_file, request, Flask
from sklearn.externals.six import StringIO
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/get_image', methods=['GET', 'POST'])
def get_image():
    if os.path.exists("foo.png"):
        os.remove("foo.png")
    warnings.filterwarnings('ignore')
    var_indep_cat = ['COLLEGE', 'REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL',
                     'CONSIDERING_CHANGE_OF_PLAN', 'LEAVE']

    var_indep_num = ['INCOME', 'OVERAGE', 'LEFTOVER', 'HOUSE', 'HANDSET_PRICE',
                     'OVER_15MINS_CALLS_PER_MONTH', 'AVERAGE_CALL_DURATION']
    var_indep_cat.remove('LEAVE')

    csvRest = request.files['file']
    print("Received csv: " + csvRest.filename)
    csvRest.save("churn.csv")
    choosenvar = request.args.get('var')
    data = pd.read_csv('churn.csv', sep=';', na_values=".")

    isCat = any(choosenvar in x for x in var_indep_cat)
    if isCat:
        sns.countplot(x=choosenvar, hue="LEAVE", data=data)
        plt.title(choosenvar + ' Count Plot')
        plt.legend(['LEAVE', 'STAY'])
        plt.savefig('foo.png')
        filename = 'foo.png'
        plt.clf()
    else:
        sns.kdeplot(data.loc[(data['LEAVE'] == 'LEAVE'),
                             choosenvar], color='r', shade=True, Label='LEAVE')
        sns.kdeplot(data.loc[(data['LEAVE'] == 'STAY'),
                             choosenvar], color='b', shade=True, Label='STAY')
        plt.title('Graphic')
        plt.xlabel(choosenvar)
        plt.ylabel('Probability Leave')
        plt.savefig('foo.png')

        filename = 'foo.png'
        plt.clf()

    return send_file(filename, mimetype='foo.png')


@app.route('/get_tree', methods=['GET', 'POST'])
def get_tree():
    csvRest = request.files['file']
    csvName = werkzeug.utils.secure_filename(csvRest.filename)
    print("Received csv: " + csvRest.filename)
    csvRest.save("02_churn.csv")
    choosenLevel = int(request.args.get('level'))
    data = pd.read_csv('02_churn.csv', sep=';', na_values=".")

    var_indep_cat = ['COLLEGE', 'REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL',
                     'CONSIDERING_CHANGE_OF_PLAN', 'LEAVE']

    var_indep_num = ['INCOME', 'OVERAGE', 'LEFTOVER', 'HOUSE', 'HANDSET_PRICE',
                     'OVER_15MINS_CALLS_PER_MONTH', 'AVERAGE_CALL_DURATION']

    var_indep_cat.remove('LEAVE')

    data_cat_one_hot = pd.get_dummies(data[var_indep_cat], prefix=var_indep_cat)
    X = data[var_indep_num].join(data_cat_one_hot)
    y = data['LEAVE']
    X.shape

    np.random.seed(1234)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    import warnings
    warnings.filterwarnings('ignore')

    np.random.seed(1234)
    ctree = DecisionTreeClassifier(
        criterion='entropy',  # el criterio de particionamiento de un conjunto de datos
        max_depth=choosenLevel,  # prepoda: controla la profundidad del árbol (largo máximo de las ramas)
        min_samples_split=3,  # prepoda: el mínimo número de registros necesarios para crear una nueva rama
        min_samples_leaf=2,  # prepoda: el mínimo número de registros en una hoja
        random_state=None,  # semilla del generador aleatorio utilizado para
        max_leaf_nodes=12,  # prepoda: máximo número de nodos hojas
        min_impurity_decrease=0.0,
        # prepoda: umbral mínimo de reducción de la impureza para aceptar la creación de una rama
        class_weight=None  # permite asociar pesos a las clases, en el caso de diferencias de importancia entre ellas
    )
    ctree.fit(X_train, y_train)

    dot_data = StringIO()
    tree.export_graphviz(ctree,
                         filled=True, rounded=True,  # nodos redondeados y coloreados
                         class_names=ctree.classes_,
                         feature_names=X_train.columns,
                         out_file=dot_data,
                         special_characters=True,
                         proportion=True,
                         impurity=True,
                         precision=2
                         )
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())
    graph.write_png("tree.png")

    filename = "tree.png"
    return send_file(filename, mimetype='foo.png')


@app.route('/get_data_tree', methods=['GET', 'POST'])
def get_data_tree():
    csvRest = request.files['file']
    csvName = werkzeug.utils.secure_filename(csvRest.filename)
    print("Received csv: " + csvRest.filename)
    csvRest.save("02_churn.csv")
    choosenLevel = int(request.args.get('level'))
    data = pd.read_csv('02_churn.csv', sep=';', na_values=".")

    var_indep_cat = ['COLLEGE', 'REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL',
                     'CONSIDERING_CHANGE_OF_PLAN', 'LEAVE']

    var_indep_num = ['INCOME', 'OVERAGE', 'LEFTOVER', 'HOUSE', 'HANDSET_PRICE',
                     'OVER_15MINS_CALLS_PER_MONTH', 'AVERAGE_CALL_DURATION']

    var_indep_cat.remove('LEAVE')

    data_cat_one_hot = pd.get_dummies(data[var_indep_cat], prefix=var_indep_cat)
    X = data[var_indep_num].join(data_cat_one_hot)
    y = data['LEAVE']
    X.shape

    np.random.seed(1234)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    import warnings
    warnings.filterwarnings('ignore')

    np.random.seed(1234)
    ctree = DecisionTreeClassifier(
        criterion='entropy',  # el criterio de particionamiento de un conjunto de datos
        max_depth=choosenLevel,  # prepoda: controla la profundidad del árbol (largo máximo de las ramas)
        min_samples_split=3,  # prepoda: el mínimo número de registros necesarios para crear una nueva rama
        min_samples_leaf=2,  # prepoda: el mínimo número de registros en una hoja
        random_state=None,  # semilla del generador aleatorio utilizado para
        max_leaf_nodes=12,  # prepoda: máximo número de nodos hojas
        min_impurity_decrease=0.0,
        # prepoda: umbral mínimo de reducción de la impureza para aceptar la creación de una rama
        class_weight=None  # permite asociar pesos a las clases, en el caso de diferencias de importancia entre ellas
    )
    ctree.fit(X_train, y_train)

    y_pred = ctree.predict(X_test)
    accuracy = 'Accuracy: ' + str(metrics.accuracy_score(y_test, y_pred))
    kappa = 'Kappa: ' + str(metrics.cohen_kappa_score(y_test, y_pred))

    return '{} , {}'.format(accuracy, kappa)

#@app.route('/get_scatter', methods=['GET', 'POST'])
#def get_scatter():
    


app.run(host='0.0.0.0', port=8082, debug=True)
