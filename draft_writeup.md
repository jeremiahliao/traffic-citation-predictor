Group Members:
Jeremiah Liao
Kevin Diaz (kjdiaz@calpoly.edu)

For our initial draft we focused on getting down some baseline classifiers and a draft for how we will start to pick and engineer features. The goal of this classifier is to take in the features of a traffic stop and determine if it resulted in a citation or a warning.

We compared a random baseline, an "always citation" model, an "always warning" model, and a random forest model.

This is a table of our results.
| Model               | Accuracy  | F1 Score                 | ROC AUC   |
| ------------------- | --------- | ------------------------ | --------- |
| **Random**          | 0.498     | 0.506                    | 0.500     |
| **Always Citation** | 0.515     | 0.680 (for class 1 only) | 0.500     |
| **Always Warning**  | 0.485     | 0.000                    | 0.500     |
| **Random Forest**   | **0.729** | **0.710**                | **0.806** |

The dataset has around a 52% citation rate, so always predicting all citations gets around 50% accuracy. It has an f1 score of 0.680, which is due to how balanced the classes are. The model will have a 100% recall and around a 50% precision by simply predicting citation for every traffic stop, but this is only for the citation class. It can never predict warnings, so the f1 score is slightly misleading

Our initial random forest model on the other hand does seem to have considerable discriminative power. We are using the following features:
