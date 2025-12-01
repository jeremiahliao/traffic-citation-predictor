Jeremiah Liao
Kevin Diaz (kjdiaz@calpoly.edu)
# Predicting Traffic Citation Outcomes From Traffic Stop Data
## 1. Introduction

In this project, we investigate whether we can classify the outcome of a traffic stop as either a citation or a warning using only the administrative fields recorded by the officer during the stop. We want to know how well we can classify these outcomes, as well as how factors such as race or gender affect the model's effectiveness. We belive that through this analysis we can gain a better understanding of any unfair biases present in the data.

Our goal is to use the predictive performance of our model based on each feature used during training to determine which factors contribute the most to an officer's decision to issue a citation. Specifically, we compare a simple charge-based heuristic to a gradient-boosted model that uses a richer set of features, and then study what happens when contextual features such as race and gender are added or removed. Our results show that most of the model's predictive ability comes from the charge which the officer chose to cite/warn under. Additionally, a small number of contextual fields such as the time of day, day of the week, and location (closely tied to police agency) also have an minor, but noticeable, effect. Race and gender on the other hand seem to have only a marginal impact in overall metrics despite notable disparities in citation rates across demographics.

## 2. Data

### 2.1 Source and scope

The dataset we use originally comes from a publically released dataset of electronically logged traffic violations in Montgomery County, Maryland, and nearby areas. This dataset was previously hosted on data.gov, where data up until 2023 was available. However, it appears it has since been take down, so we work with a snapshot of this dataset from roughly 8 years ago available on kaggle. The data in this set spans from the beginning of 2012 to the end of 2016.

The original csv file contained 1,018,634 records. The data was already very clean, with very few missing values. An issue we encountered was the vehicle make and model, as well as the charge description. The `Make` column contained all sorts of abbreviations and misspellings of common makes. We were able to use fuzzy matching to narrow this down to 63 different makes, with under three thousand records left as 'unknown'. The description column seemed to be possibly useful, but upon closer examination it seems to closely correlate with charge, with only minor differences among descriptions for the same charge.

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

We also include some minor boolean columns
- `Personal Injury`, `Commercial License`, `Alcohol`. `Property Damage`, etc.

However, these columns appear to be problematic. Only a small fraction of all rows contain a true value for any column, despite the issued charge suggesting that they should be true. When included, these columns made almost no difference in model performance.

### 