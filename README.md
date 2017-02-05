# Apprentissage avec un model linéaire

# Introduction
Dans ce tutoriel nous allons montrer un exemple d'utilisation du tensorflow de google à travers un exemple d'application sur l'apprentissage de données à travers un model linéaire de graphe. La base de données considérée ici est le MNIST.
> N'hésitez pas à lire les commentaires entre les lignes de codes. C'est surtout là que vous trouverez les bonnes explications

# L'apprentissage
**1. Importation des librairies**

Nous allons tout d'abord commencer par importer les bibliothèques utiles pour notre programme

```Python
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
```
	
> Notez que selon la façon dont vos bibliothèques sont rangées, le chemin spécifié pour importer **input_data** peut ne pas être exact. Donc pensez à changer selon la structuration de vos packages.

        
**2. Définition de quelques paramètres importants**
```Python
learning_rate = 0.7
training_iteration = 30
batch_size = 100
display_step = 2
img_size = 28 #la taille des images dans MNIST est de 28 pixels
img_size_flat = img_size*img_size #28x28 pixels, la matrice 2D de l'image sera transformée en un vecteur de taille 784
img_shape = (img_size,img_size) #qui sera réutilisé plus tard pour convertir le vecteur en image
num_classes = 10 #10 classes possibles 0-9
```

Les images de la base de données MNIST sont ensuite récupérées. Elles seront téléchargées si elles n'existent pas déjà dans le répertoire que vous spécifierez; ici "/tmp/tensorflow"
```Python
mnist = input_data.read_data_sets("/tmp/tensorflow",one_hot=True)
```

la base de données mnist est divisé en 3 sous-ensembles:
- **mnist**
    - **mnist.train :** la base d'apprentissage contenant elle même
        - **mnist.train.images :** les images de la base
        - **mnist.train.labels :** les vrais labels de ces images codés en **one-hot** c-à-d que le label 7 par exemple 				s'écrira [0,0,0,0,0,0,0,1,0,0] un vecteur de taille 10 avec un "1" à la 7 case pour dire que le label 				vaut 7

    - **mnist.test :** la base de test
        - **mnist.tests.images :** les images de la base
        - **mnist.tests.labels :** les vrais labels

    - **mnist.validation :** la bas de validation
        - **mnist.train.images :** les images de la base
        - **mnist.train.labels :** les vrais labels

Le codage en **one-hot** est utile pour le travail à venir mais nous aurons également besoin de la vraie valeur du label. Pour cela nous allons ajouter un autre champ cls (pour dire class) au champ test qui contient les labels sous forme de nombre. On prend ainsi chaque label (par exemple [0,0,0,0,0,0,1,0,0,0]) et on convertit avec argmax() ce qui donne pour notre exemple 6 (le "1" se trouve à la 6e case).
>argmax() retourne l'indice du maximum du vecteur.

```Python
mnist.test.cls = np.array([label.argmax() for label in mnist.test.labels])
### mnist.test.labels[0:3,:] #décommenté si vous voulez voir le contenu
### mnist.test.cls[0:3] #décommenté si vous voulez voir le contenu de la conversion et comparé les 2
```
>Notez ici que mnist.test.labels est une matrice de taille 55000x10 où 55000 correspond au nombre d'image et 10 faisant référence au codage one-hot alors que mnist.test.cls est un vecteur de taille 55000 (plus de one-hot)

**3. Variables d'entrée du tensorflow**

**x** représente les images que l'on présente à l'entrée du graphe (du tensorflow) et **y** la sortie
```Python
x = tf.placeholder("float",[None, img_size_flat]) #img_size_flat = 28*28 = 784
```
j'ai déjà expliqué que chaque image 2D était convertit en un vecteur de taille img_size_flat. **None** veut dire qu'à priori on ne connait pas le nombre d'image à l'entrée du tenseur. x est donc une matrice de Nbre_lign(inconnu) lignes (le nombre d'images) et de 784 colonnes.
>Par inconnu comprenez "dont la taille peut changer en fonction la quantité de données à l'entrée du tenseur"

```Python
y = tf.placeholder("float",[None,10]) #la sortie ne peut être que parmi 0-9 digits ==> 10 classes
```
Comme on ne connait pas le nombre d'images à l'entrée on ne connait pas non plus le nombre de labels correspondant à la sortie  mais on sait à fortiori que ces labels sont entre 0 et 9. y est donc une matrice de taille Nbre_lign (inconnu) lignes et 10 colonnes (10 fait référence au codage one-hot)
```Python
y_cls = tf.placeholder(tf.int64,[None]) #ici y_cls est un vecteur de taille Nbre_lign(le nombre d'image) et dont chaque élément contiendra finalement le label de la classe prédite pour chaque image (plus de one-hot) d'où le "tf.int64" qui spécifie la nature de chaque élément du vecteur.
```

	     im1 ......... ################# ------>     y1
	     im2 ......... #               # ------>     y2
	     im3 ......... #               # ------>      .
	 x = .             #  graphe(W,b)  # ------> y =  .
	     .             #               # ------>
	     .             #               # ------>      .
	     imn ......... ################# ------>      yn


**4. création du model linéaire**

On va utiliser ici un model linéaire pour apprendre à notre graphe. Autrement dit y = W*x+b ou  W désigne le poids et b le biais.
```Python
W = tf.Variable(tf.zeros([img_size_flat,num_classes])) # matrice de taille 784x10
```

**W*x** est une matrice de Nbre_lign lignes et de num_classes ( = 10) colonnes.
```Python
b = tf.Variable(tf.zeros([num_classes]))
```
Le biais est donc naturellement un vecteur ligne de taille num_classes. on va ajouter un biais à chaque colonne de W*x ie on ajoute b[i] à tous les éléments de la colonne i de la matrice W*x

	# ceci est un exemple de la façon dont la somme se fait
	>>> u
	array([[1, 2],
	       [3, 4]])
	>>> u+np.array([-1,-2])
	array([[0, 0],
	       [2, 2]])
	# en sommant on ajoute le vecteur np.array([-1,-2]) à chaque ligne

/!\ notons ici qu'on ne fait que définir nos variables et notre model. Tout est initialisé à zéros; on a encore rien fait . On est simplement en train de juste supposer que la sortie dépend de l'entrée de façon linéaire. Ce qu'on va maintenant chercher à faire, c'est trouver le W et le b qui rendent au mieux cette supposition vrai. Pour ça, on va se servir des données que nous avons mnist.train qui contient des images dont on connait les labels pour ensuite chercher par optimisation le W et le b qui va au mieux.
```Python
with tf.name_scope("Wx_b") as scope: #cette ligne c'est pour utiliser plus tard le tensorboard
    #construction du model linéaire y = W*x+b
    model = tf.matmul(x,W)+b
    #ici model est une matrice de None(nombre d'image) lignes et de 10 colonnes ou chaque élément de
    #la i ème ligne et la j ème colonne donne une estimation d'à quel point l'image i est susceptible
    #d'appartenir à la classe j
    # par exemple si y[0] = [0,0,1.25,7.87,23.5,0,0,0.012,0,0] alors l'image im1 a plus de chance d'être "4"

    #notons qu'ici les valeur dans y peuvent être supérieur à 1. ça serait bien de pouvoir normaliser tout ça
    #pour obtenir des valeurs entre 0 et 1 telles que sum(y(i]) = 1 (valeurs de probabilités)
    #pour ça tensorflow a un outil qui s'appelle softmax et qui va le faire pour nous

    y_pred = tf.nn.softmax(model) #y_pred le y prédite normalisée par régression softmax
    y_pred_cls = tf.argmax(y_pred,dimension=1) #on calcule la classe réelle en prenant l'indice du maximum
    #de chaque ligne. Notez que y_pred_cls est un vecteur colonne en ce moment de taille le nombre d'images.

#more name scopes will clean up graph representation
with tf.name_scope("cost_function") as scope:
    #on va essayer maintenant de minimiser l'erreur entre y_pred (la sortie prédite et la vraie
    #valeur de la sortie qui est y(définie précédemment)
    #on pourrait imaginer calculer la fonction coût comme  = sum(|y_pred - y|) (la somme des écarts entre la
    #valeur réelle et la valeur prédite: c'est légitime! pourquoi pas même l'élever au carré pour prendre l'écart
    #quadratique= erreur quadratique moyenne EQM =MSE).
    #ce qu'il faut savoir c'est que selon la méthode utilisée pour calculer l'erreur quadratique, il faut pouvoir      
    #choisir la fonction d'optimisation qui convient au mieux.
    #on va calculer l'erreur autrement en utilisant un calcul d'entropie

    ### cost_function = -tf.reduce_sum(y*tf.log(y_pred)) #calcul d'entropie comme en théorie de l'information
    #et le coût est minimal si y coïncide avec y_pred
    #seulement ici le calcul avec le log pose problème si y_pred contient des valeurs nulles
    #pour éviter cela tensorflow a un outil qui s'appelle tf.nn.softmax_cross_entropy_with_logits()

    #on écrira plutôt
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y)) #il prend en
    #paramètre le model et non y_pred(le calcul du softmax se fait à l'intérieur) et y le vrai label
    #notez que ça soit tf.reduce_sum ou tf.reduce_mean on s'en fout les 2 sont proportionnels
    #tf.reduce_mean = tf.reduce_sum/nombre_images

    ### cost_function = tf.reduce_mean(tf.square(y_pred-y)) #EQM (vous pouvez l'essayer aussi ça marche plutôt bien)
    
    #create a summary to monitor cost function
    tf.summary.scalar("cos_fuction",cost_function) #pour le tensorboard

with tf.name_scope("train") as scope:
    #optimisation par la méthode de la descente du gradient très pratique surtout pour minimiser les erreurs
    #quadratiques ici c'est à pas fixe (learning_rate = 0.5)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
```
* mesure de performances
```Python
predictions = tf.equal(y_pred_cls,y_cls)
accuracy = tf.reduce_mean(tf.cast(predictions,"float"))
```
predictions est un vecteur de boolean qui contient True à l'indice i si y_pred[i] = y_cls[i] sinon False.
ex: predictions= [False, True, False, True, True, True, True, False, True] pour 9 images
puis on calcule l'accuracy (l'exactitude) ie combien de vraie au total en % en castant les booléan en float. Avec l'ex précédent, tf.cast(predictions,"float") = [0,1,0,1,1,1,1,0,1] soit une moyenne = 6/9


Rappelons que dans tout ce qui précède on a fait que définir nos variables notre model de graphe et d'optimisation maintenant nous allons donner vie à tout cela

**5. initialisation des variables**

	init = tf.global_variables_initializer() # ou tf.initialize_all_variables() qui est obsolète pour cette version

merge all summaries into a single operator for tensorboard
```Python
#merge all summaries into a single operator for tensorboard
merged_summary_op = tf.summary.merge_all() #or tf.merge_all_summaries() but it is deprecated

#on va maintenant ouvrir une session pour le graphe
with tf.Session() as sess:
    sess.run(init)
    #Set the logs writer to the folder /tmp/tensorflow/logs
    summary_writer = tf.summary.FileWriter('/tmp/tensorflow/logs',graph=sess.graph)#tf.train.SummaryWriter
    #on crée des logs pour utiliser dans le tensorboard plus tard
    #train cycle
    for iteration in range(training_iteration):
        total_batch = int(mnist.train.num_examples/batch_size)
        #loop over all batches
        avg_cost = optimizefor(total_batch,iteration)
        #display logs per iteration step
        if iteration % display_step == 0:
            print("Iteration:",'%04d' % (iteration +1), "cost=", "{:.9f}".format(avg_cost))

        
    print("Tuning completed!")
    #on fait passer la base de test pour voir a quel point on est bon!
    print_accuracy(mnist.test,sess)
    feed_dict_test = {x:mnist.test.images,y:mnist.test.labels,y_cls:mnist.test.cls}
    print_unmatched(sess,feed_dict_test,mnist.test)
    print_matched(sess,feed_dict_test,mnist.test)
    plot_weights(sess)
    plt.show()
    plt.close()
```	
	
**6. Quelques fonctions intéressantes**
```Python
def plot_images(images, cls_true,cls_pred=None):
    assert len(images) == len(cls_true) == 9 #on affiche que 9 images: modifiable
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        #Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        #Show true and predicted classes.
        if(cls_pred is None):
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
	# Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

#la base de donnée training est très grande(55000 images) ça prendra du temps d'optimiser tout cela.On décide donc
#d'optimiser le graphe par paquet de batch_size images.   
def optimizefor(num_of_iteration,iteration):
    avg_cost = 0
    for i in range(num_of_iteration):
        x_batch,y_batch = mnist.train.next_batch(batch_size)
        #on récupère le paquet d'image pour x_batch (le lot x de batch_size image)et les labels correspondant y_batch
        #on crée un dictionnaire python
        feed_dict_train = {x:x_batch, y:y_batch} #ça veut dire que le champ feed_dict_train.x = x_batch et
        #feed_dict_train.y = y_batch
        #autrement dit le x et y définis au début seront remplis avec ces données

        #puis on appelle sess.run() pour faire tourner notre graphe (sess = tf.Session())
        sess.run(optimizer,feed_dict=feed_dict_train)
        #compute the average loss
        avg_cost += sess.run(cost_function,feed_dict=feed_dict_train)/num_of_iteration
        #write logs for each iteration
        summary_str = sess.run(merged_summary_op,feed_dict=feed_dict_train)
        summary_writer.add_summary(summary_str,iteration+num_of_iteration+i)
    return avg_cost


def print_accuracy(data,session):
    feed_dict = {x:data.images, y:data.labels, y_cls:data.cls}
    acc = session.run(accuracy,feed_dict=feed_dict)
    print("Accuracy:{0:.1%}".format(acc))

def print_unmatched(session,feed_dict,data):
    predict,predict_cls = session.run([predictions,y_pred_cls],feed_dict = feed_dict)
    #predict rappelons le contient des valeurs true ou false indiquant les prédictions qui matchent ou pas
    #pour récupérer les images qui ne matchent pas il suffit de répérer les éléments dont la valeur est False
    false = (predict == False) #une manière chouette de récupérer les valeurs False de predict
    images = data.images[false] # images contient les images qu'on a mal détecter
    cls_pred = predict_cls[false]
    cls_true = data.cls[false]
    plot_images(images[0:9],cls_true[0:9],cls_pred[0:9])

def print_matched(session,feed_dict,data):
    predict,predict_cls = session.run([predictions,y_pred_cls],feed_dict = feed_dict)
    #predict rappelons le contient des valeurs true ou false indiquant les prédictions qui matchent ou pas
    #pour récupérer les images qui ne matchent pas il suffit de répérer les éléments dont la valeur est False
    true = (predict == True) #une manière chouette de récupérer les valeurs False de predict
    images = data.images[true] # images contient les images qu'on a mal détecter
    cls_pred = predict_cls[true]
    cls_true = data.cls[true]
    plot_images(images[0:9],cls_true[0:9],cls_pred[0:9])

def plot_weights(session):
    w = session.run(W) #W c'est la matrice des poids de taille 784x10
    wmin = np.min(w) #le minimum de touts les éléments de la matrice
    wmax = np.max(w) # le  maximum de touts les éléments de la matrice
    fig,axes = plt.subplots(3,4)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    for i,ax in enumerate(axes.flat):
        if i<10:
            image = w[:,i].reshape(img_shape) #w[:,i] toute la i eme colonne en vecteur refaçonné en image 28x28
            ax.set_xlabel("weights: {0}".format(i))
            ax.imshow(image,vmin=wmin,vmax=wmax,cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
```
