Jeremiah Liao (jjliao@calpoly.edu)
Kevin Diaz (kjdiaz@calpoly.edu)
# Predicting Traffic Citation Outcomes From Traffic Stop Data
## 1. Introduction

In this project, we investigate whether we can classify the outcome of a traffic stop as either a citation or a warning using only the administrative fields recorded by the officer during the stop. We want to know how well we can classify these outcomes, as well as how factors such as race or gender affect the model's effectiveness. We believe that through this analysis we can gain a better understanding of any unfair biases present in the data.

Our goal is to use the predictive performance of our model based on each feature used during training to determine which factors contribute the most to an officer's decision to issue a citation. Specifically, we compare a simple charge-based heuristic to a gradient-boosted model that uses a richer set of features, and then study what happens when contextual features such as race and gender are added or removed. Our results show that most of the model's predictive ability comes from the charge which the officer chose to cite/warn under. Additionally, a small number of contextual fields such as the time of day, day of the week, and location (closely tied to police agency) also have an minor, but noticeable, effect. Race and gender on the other hand seem to have only a marginal impact in overall metrics despite notable disparities in citation rates across demographics.

## 2. Data

### 2.1 Source and scope

The dataset we use originally comes from a publically released dataset of electronically logged traffic violations in Montgomery County, Maryland, and nearby areas. This dataset was previously hosted on data.gov, where data up until 2023 was available. However, it appears it has since been take down, so we work with a snapshot of this dataset from roughly 8 years ago available on kaggle. The data in this set spans from the beginning of 2012 to the end of 2016.

The original csv file contained 1,018,634 records. The data was already very clean, with very few missing values. For most missing values, we either filled it with an 'Unknown' label, imputed the value using other columns or a mean value. An issue we encountered was the vehicle make and model, as well as the charge description. The `Make` column contained all sorts of abbreviations and misspellings of common makes. We were able to use fuzzy matching to narrow this down to 63 different makes, with under three thousand records left as 'unknown'. The description column seemed to be possibly useful, but upon closer examination it seems to closely correlate with charge, with only minor differences among descriptions for the same charge. The `VehicleType` column had some minor data errors with some duplicate values referring to the same vehicle type. (For example, "18 - Police Vehicle" and "18 - Police (Non-Emerg)" referred to the same vehicle type). To fix this, we extracted the numerical code referring to the vehicle type and remapped it to a standard vehicle type name.

By restricting to only stops which resulted in a citation or a warning, and then filtering to the top 255 most common charges (out of almost 900) we are left with 954,440 rows in our dataset after cleaning. We focused on only the 255 most common charges because there were very few entries for the rest of the charges comparatively, and because our model (histogram gradient boosting) works most efficiently when categorical features are limited to 255 dimensions due to internal implementation details.
j
### 2.2 Features used

The target label is `Violation Type`, with a 0 representing warning and a 1 representing citation.

We use the following features:

- `Charge`: the traffic article violation as determined by the officer
- `Arrest Type`: How the violation was identified or the stop initiated (e.g, marked patrol, license plate recognition, laser, etc).
- `SubAgency`: The district which the police officer belongs to
- `VehicleType`: Automobile, motorcycle, bus, RV, etc.
- `Race`, `Gender`: Demographic information
- `dayofweek`, `hour`: the day of the week and the time of the stop.
- `Latitude`, `Longitude`: The precise geolocation of the stop.

We include features such as charge, location, and time of day, as these features indicate the severity of the violation and danger it poses to other drivers on the road. DUI charges will likely have a higher likelhood of a citation issued compared to running a stop sign because it presents a greater danger to other drivers. (This was also reflected in our exploratory data analysis). While spatial and temporal features may indicate traffic conditions at the time of the violation. We hypothesize that the greater the danger, the higher the chances a citation is issued.

We include driver specific features such as race, gender, vehicle type to see if there are any underlying bias from the police officer. Which subagency the officer is from may also reveal underlying culture or biases present in a specific station.  

We also include some boolean columns
- `Personal Injury`, `Commercial License`, `Alcohol`. `Property Damage`, etc.

These columns give further insight into the traffic incident, such as whether the driver had alcohol or contributed to an accident. 

Upon initial glance, these columns appear to be irrelevant. Only a small fraction of all rows contain a true value for any column, with some seemingly contradicting the issued charge. For example, most DUI charges do not have the `Alcohol` field marked, while a non-DUI charge such as speeding may have that field marked. However, upon further analysis, the presence of such fields, even if rare, does almost always indicate a citation. Still, since they are so rare they had almost no impact on model performance.

## 3. Model

We compared 3 models for classifying the outcome of a traffic stop - a baseline model, a heuristic model, and a historgram gradient boosting classifier.

### 3.1 Baseline Model

Our baseline model is a classifier that always predicts that a traffic stop will result in a citation issued. 

We use this as our baseline to see how well more complex models can identify the instances where a warning may be given for a traffic stop.

### 3.2 Heuristic Model

This model uses the mean citation rate for a given charge as the probabiity of issuing a citation, with the treshold set at 0.5.

During our intial analysis, we identified that the citation rates varied by the charge corresponding to the traffic stop.


## 4. Evaluation & Analysis

## 5. Conclusion

