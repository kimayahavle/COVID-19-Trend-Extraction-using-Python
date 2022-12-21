# COVID-19 Trend Extraction using Python
Description: This project emplyoys BeautifulSoup to scrape the COVID-19 data from the Worldometer website. Further, it utilizes the Pandas, Seaborn, Matplotlib modules to process the tabluar data and visualize the charts for COVID-19 trends. The Scikit learn module then performs linear regression prediction.

### How to run
* Run the notebook.py file from the terminal using `python notebook.py`
* The program will scrape data from https://www.worldometers.info/coronavirus/ and store the CSV file in the `dataset` directory
* It will generate graphs and store them in the `graphs` directory

### Dependencies
* BeautifulSoup 4.11.1
* numpy 1.21.5
* pandas 1.4.2
* matplotlib 3.5.1
* seaborn 0.11.2
* scikit-learn 1.0.2