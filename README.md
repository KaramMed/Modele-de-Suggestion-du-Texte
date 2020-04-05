# Modele-de-Suggestion-du-Texte
Modele de suggestion du texte manquant avec un Reseau de Neurones Recurrents avec cellules LSTM
Problématique : determiner le mot suivant d'une séquences de mots
- Extraction de données : on va utiliser comme source de données , un livre de Friedrich Nietzsche "Beyond Good and Evil " 
- Processus NLP : afin de traiter du texte , premièrement on va utiliser chaque caractere au lieu de tout le mot , cela va nous rendre un vocabulaire moins chargé pour l'entrainement , ainsi il va falloir convertir chaque caractere en un nombre pour qu’il soit traité par le modele
- Preparation de données : on doit enlever les caracteres speciaux et les nombres et unifier la casse pour nettoyer les données et avoir un petit vocabulaire 
La preparation sera faite de la manière suivante : 
Phrase : « Bonjour ! »
Preparation : « Bonjour ! »  « bonjour »
Transformation NLP : « bonjour »  « 2,15,14,10,15,21,18 »
Normalisation : on utilise un vecteur One-Hot pour la sequence de nombre afin de normaliser l’ecart entre les données
Le vecteur sera de la taille du vocabulaire , et aura la valeur de « 1 » dans l’endroit du nombre de la lettre
- Entrainement du modele : il va falloir utiliser un reseaux de neurones a convolution , pour la simple raison de garder une memoire pendant la prediction , exemple : imaginons les 2 phrases suivantes " il fait " et " fait " , il faut qu'on prédit des resultats differents selon ce qui se trouve avant le mot " fait " et pas la même chose

L’entrainement a travers les données issues du livre sera faite d'une maniere que chaque lettre est l'entrée et prends comme cible
le mot suivant 
Exemple : le mot " bon " --> (entrée : b , cible : o) , (entrée : o , cible : n) , (entrée : n , cible : )

-----------------------------------------------

# Text-Suggestion-Model
Suggestion model for missing text with a Recurrent Neural Network with LSTM cells
Problem: determine the next word in a sequence of words
- Data extraction: we will use as a data source, a book by Friedrich Nietzsche "Beyond Good and Evil"
- NLP process: in order to process the text, first we will use each character instead of the whole word, this will make our vocabulary less busy for training, so we will have to convert each character into a number so that it be treated by the model
- Data preparation: we must remove special characters and numbers and unify the case to clean the data and have a small vocabulary
The preparation will be done as follows:
Phrase: "Hello! "
Preparation: "Hello! "" hello "
NLP transformation: "hello"  "2,15,14,10,15,21,18"
Normalization: we use a One-Hot vector for the number sequence in order to normalize the difference between the data
The vector will be the size of the vocabulary, and will have the value of "1" in the place of the number of the letter
- Model training: we will have to use a convolutional neural network, for the simple reason of keeping a memory during prediction, example: imagine the following 2 sentences "it does" and "does", we have to predict different results depending on what is before the word "done" and not the same thing

The training through the data from the book will be done in a way that each letter is the entry and target
the next word
Example: the word "bon" -> (entry: b, target: o), (entry: o, target: n), (entry: n, target:)























