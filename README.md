# aai-501-bike-share-prediction
AAI-501 Group 3 Final Project: Bike Share Demand Prediction and Rider Segmentation

# Bike Share Demand Prediction

This project is a part of the AAI-501 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

**Project Status:** Completed

## Project Intro/Objective
The main purpose of this project is to predict bike share rental demand and group rider patterns using weather and calendar data. By finding out when rentals are lowest, bike share companies can schedule major bike repairs when it bothers the fewest riders.

## Partner(s)/Contributor(s)
* Buffy Barentine – Data Cleanup and Preparation
* Angelo Walsh – Machine Learning and Coding 
* James Arenas – Data Analysis and Report 

## Methods Used
* Data Manipulation & Cleanup
* Machine Learning (Linear Regression & K-Means Clustering)
* Data Visualization

## Technologies
* Python (following PEP 8 style rules)
* Pandas & NumPy
* Scikit-Learn
* Matplotlib & Seaborn
* Git & GitHub

## Project Description
We are using the UCI Bike Sharing dataset (`hour.csv`), which has 17,379 hourly records and 17 columns from Capital Bikeshare (2011–2012). 

**Key Variables:**
* `season`, `yr`, `mnth`, `hr`, `holiday`, `workingday`: Date and time details.
* `weathersit`, `temp`, `atemp`, `hum`, `windspeed`: Weather conditions.
* `casual`, `registered`: Rentals by non-members vs. members.
* `cnt`: Total bike rentals (our main target).

**Questions We Are Exploring:**
1. How well can Linear Regression predict total hourly bike rentals?
2. Do casual and registered riders fall into different groups based on time and weather?
3. What days, hours, or seasons have the lowest demand for scheduling repairs?

**Challenges:**
We must prevent data leakage by keeping `casual` and `registered` out of our regression inputs, scaling data before clustering, and splitting training and testing data in time order.

## Installation
1. Clone this repository: `git clone https://github.com/buffybarentine/aai-501-bike-share-prediction`
2. Install the required Python libraries: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run the scripts in order: `python data_preprocessing.py`, then `python train_models.py`, then `python evaluate_results.py`

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
Thank you to Professor Friesen and to Professor Van Benschoten at USD for guidance, and to Hadi Fanaee-T and Joao Gama for providing the UCI Bike Sharing dataset.
