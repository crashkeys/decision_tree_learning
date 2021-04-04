# decision_tree_learning
#################
NOTES ON THE CODE
#################

Il file su cui eseguire il test è "TestDataset.py", in cui è possibile fare il test sui tre dataset scelti uno alla volta e modificare a piacere il numero di folds (purché rientri nei limiti della dimensione delle righe del dataset). I dataset sono in formato .csv e importati nel programma tramite le funzioni descritte in "Helpers.py", specifiche per ogni tipo di dataset (categorico, numerico o misto).

Il file "DecisionTreeLearning.py" contiene la funzione stessa per la costruzione dell'albero di decisione.

Per il progetto sono state usate le librerie math, numpy, StratifiedKfold di sklearn, groupby di intertools, re, time e csv.

Per una stampa più comprensibile dell'albero e la visualizzazione dei risultati sono stati usati i colori tramite la libreria colorama.


Il codice è stato scritto in Python 3.8

#############
FONTI ESTERNE
#############
- I file "DecisionLeaf.py" e "DecisionNode.py" e la funzione __call__ di quest'ultimo per testare un esempio del dataset sono ispirati al codice di dmeoli (https://github.com/aimacode/aima-python/blob/master/learning.py)

- La funzione "find-threshold" è presa da Sanjay-Chivukula (https://github.com/sanjay-chivukula/stackoverflow-projects/blob/main/break-sequence/break_sequence.py) 


#############################################################
Datasets infos from the originals datasets description on UCI
#############################################################

Se necessario, i dataset sono stati modificati in modo da avere "Goal" sull'ultima colonna. Tutti i dataset considerati hanno output binario e non presentano dati mancanti.

###########
TIC-TAC-TOE
###########

This database encodes the complete set of possible board configurations at the end of tic-tac-toe games, where "x" is assumed to have played first. The target concept is "win for x" (i.e., true when "x" has one of 8 possible ways to create a "three-in-a-row").

Number of Instances: 958 (legal tic-tac-toe endgame boards)

Number of Attributes: 9, each corresponding to one tic-tac-toe square

Attribute Information: (x=player x has taken, o=player o has taken, b=blank)
A1. top-left-square: {x,o,b}
A2. top-middle-square: {x,o,b}
A3. top-right-square: {x,o,b}
B1. middle-left-square: {x,o,b}
B2. middle-middle-square: {x,o,b}
B3. middle-right-square: {x,o,b}
C1. bottom-left-square: {x,o,b}
C2. bottom-middle-square: {x,o,b}
C3. bottom-right-square: {x,o,b}
Goal Class: {positive,negative}

########################
BREAST-CANCER-DIAGNOSTIC
########################

Number of instances: 569 

Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)

Attribute information

Ten real-valued features are computed for each cell nucleus:

	a) radius (mean of distances from center to points on the perimeter)
	b) texture (standard deviation of gray-scale values)
	c) perimeter
	d) area
	e) smoothness (local variation in radius lengths)
	f) compactness (perimeter^2 / area - 1.0)
	g) concavity (severity of concave portions of the contour)
	h) concave points (number of concave portions of the contour)
	i) symmetry 
	j) fractal dimension ("coastline approximation" - 1)
	k) Diagnosis (M = malignant, B = benign)

The mean, standard error, and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features.  For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

#######################
HEART-DISEASE-CLEVELAND
#######################

Number of Instances: 
          Cleveland: 303
          Hungarian: 294
        Switzerland: 123
      Long Beach VA: 200

Number of Attributes: 76 (including the predicted attribute)

Attribute Information:
   -- Only 14 used
      -- 1. #3  (age)       
      -- 2. #4  (sex)       
      -- 3. #9  (cp)        
      -- 4. #10 (trestbps)  
      -- 5. #12 (chol)      
      -- 6. #16 (fbs)       
      -- 7. #19 (restecg)   
      -- 8. #32 (thalach)   
      -- 9. #38 (exang)     
      -- 10. #40 (oldpeak)   
      -- 11. #41 (slope)     
      -- 12. #44 (ca)        
      -- 13. #51 (thal)      
      -- 14. #58 (goal)       (the predicted attribute)

The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).
