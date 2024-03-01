# Codes for the thesis "Exploring Dynamic Changes due to Ataxia Through Virtual Brain Models"
## by Elide Brunelli 
Supervisor Marialaura De Grazia, and Professors Claudia Casellato & Egidio D'Angelo

## Overview

The aim of this thesis is to explore brain modeling in mice with the use of The Virtual Brain, starting from subjects’ data coming from resting-state fMRI, as in Pagani et al. (2021). From these dataset, a mouse brain model is fine tuned on TVB starting from structural connectivty coming from Oh et al. (2014) to computationally reproduce the brain dynamics in healthy subjects. The second step was then to extend these investigations into an ataxic model, by varying the parameters of the model, in order to reproduce experimental evidence of ataxic changes in mice.


## Installation

The code is meant to be run using the infrastructure provided by The Virtual Brain. All required python libraries are contained in the requirements.txt file.

pip install -r requirements.txt

The mice dataset are not included in this repository due to space availability issues. All mouse dataset should work with this code.
