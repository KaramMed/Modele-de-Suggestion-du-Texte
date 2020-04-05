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

L’entrainement a travers les données issues du livre sera faite de la manière suivante :
Phrase : « Bonjour  »
Entrainement : 
Entrée	Cible
b	      o
o	      n
n	      j
j	      o
o	      u
u	      r
r	      ‘’
























