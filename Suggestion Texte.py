import numpy as np
np.random.seed(42)
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from keras.layers import TimeDistributed
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
from pylab import rcParams
import unidecode
import json,random


# on lite le texte du livre
texte = open("nietzshe.txt","r",encoding="utf-8").read()

# on va faire du preprocessing ( unifier le maj et min , les points , virgules , caracteres speciaux ... etc )
# on utilise unidecode pour enlever les accents etc
texte = unidecode.unidecode(texte) # rendre ASCII en texte unicode
texte = texte.lower() # pour unifier la casse
# enlever nombres et caracteres speciaux
# ( on peut garder espace et ' car seront utilisés )
for i in ("1","2","3","4","5","6","7","8","9") :
    texte = texte.replace(i," ")
for i in ("<",">","!","?","-","$",";","Ã","©","_","-","«","»","*") :
    texte = texte.replace(i, " ")

texte = texte.strip()
vocab = list(set(texte)) # les differents lettres dans le livre
# ce qu'on va faire ici c'est generer lettre par lettre , car le fait d utiliser des mots sera difficile


# on va mtn traduire chaque lettre en nombre entier , pour qu'elle soit traité par le reseau de neurones
vocab_to_int = {l:i for i,l in enumerate(vocab)} # pour traduire vocabulaire en nombre
int_to_vocab = {i:l for i,l in enumerate(vocab)} # pour traduire nombre en vocabulaire
# c'est des dictionnaires ( lettre : entier ou entier : lettre )

# maintenant on va representer tout le texte sous forme de nombres
texte_codé = [vocab_to_int[l] for l in texte]
texte_decodé = [int_to_vocab[i] for i in texte_codé]

# creation de chuncks
SEQUENCE_LENGTH = 40 # taille sequence
step = 3 # le pas
sentences = []
next_chars = []
for i in range(0, len(texte) - SEQUENCE_LENGTH, step):
    sentences.append(texte[i: i + SEQUENCE_LENGTH])
    next_chars.append(texte[i + SEQUENCE_LENGTH])

# convertion des caracteres sous forme de vecteurs one hot
X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(vocab)), dtype=np.bool) # l'entrée
y = np.zeros((len(sentences), len(vocab)), dtype=np.bool) # ce qui doit etre predit
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, vocab_to_int[char]] = 1
    y[i, vocab_to_int[next_chars[i]]] = 1




# creation du modele
model = Sequential()
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(vocab)))) # cellule de 128 valeurs
model.add(Dense(len(vocab))) # deuxieme couche
model.add(Dense(len(vocab))) # 2
model.add(Dense(len(vocab))) # 3
model.add(Activation('softmax')) # couche de sortie avec fonction softmax

# entrainement du modele
optimizer = RMSprop(lr=0.01) # learning rate de 0.01
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# 20 iterations , pourcentage de données de validation 5%
history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=10, shuffle=True).history

# sauvegarder le modele pour prochaine utilisation
##model.save('modele_suggestion.h5')
##pickle.dump(history, open("history.p", "wb"))

# chargement du modele sauvegardé
#model = load_model('modele_suggestion.h5')
#history = pickle.load(open("history.p", "rb"))

# Affichage de taux de precision
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Taux de precision')
plt.ylabel('precision')
plt.xlabel('iteration')
plt.legend(['entrainement', 'test'], loc='upper left');
plt.show()
# Affichage de taux d'erreur
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Taux d erreur')
plt.ylabel('erreur')
plt.xlabel('iteration')
plt.legend(['entrainement', 'test'], loc='upper left');
plt.show( )

# fonction de preparation d'entrée
def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(vocab)))
    for t, char in enumerate(text):
        x[0, t, vocab_to_int[char]] = 1.
    return x # retourne un vecteur pour une sequence texte

# cette fonction donne le nombre de caracteres predites possibles
def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)


# cette fonction prends un texte et retourne la suggestion
def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = int_to_vocab[next_index]
        text = text[1:] + next_char
        completion += next_char

        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion

# celle ci fait la prediction pour plusieurs textes
def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0] # verbose sert a faire un peu d'erreur pour ne pas avoir un copier coller
    next_indices = sample(preds, n)
    return [int_to_vocab[idx] + predict_completion(text[1:] + int_to_vocab[idx]) for idx in next_indices]


# essay de tester
phrases = [
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
    "That which does not kill us makes us stronger.",
    "I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
    "And those who were seen dancing were thought to be insane by those who could not hear the music.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!",

]
for q in phrases:
    seq = q[:40].lower()
    print(seq)
    print(predict_completions(seq,4)) # on veut 4 suggestion
    print()