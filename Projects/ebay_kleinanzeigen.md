## Exploring Ebay Kleinanzeigen car sales data.



 <div align="center">

  **Run this project using:** [STATIC PREVIEW](https://nbviewer.jupyter.org/github/ealvarezj/Data-Science-Portfolio/blob/main/Exploring_Ebay_Car_Sales_Data/Exploring_Ebay_Car_Sales_Data.ipynb) or [test](https://mybinder.org/v2/gh/ealvarezj/binder_env/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fealvarezj%252FData-Science-Portfolio%26urlpath%3Dtree%252FData-Science-Portfolio%252FExploring_Ebay_Car_Sales_Data%252FExploring_Ebay_Car_Sales_Data.ipynb%26branch%3Dmain)

</div>

<!-- ![sql](https://mybinder.org/badge_logo.svg) -->


In this project I explored the `Ebay Car Sales Dataset` from the [Ebay Kleinanzeigen](https://www.ebay-kleinanzeigen.de/) German website, a division for classified advertisements' website with sections devoted to jobs, housing, services, community service, gigs, and cars. Similar to "Craiglist" in the United States. The dataset was crawled from the website directly. 

![kleinanzeigen](/DataScience-Portfolio/images/kleinanzeigen.png)

The analysis was done completely using python, specially using the `pandas` library for data manipulation. Some goals of this project where:

- Data cleaning:
  - Rename features in the dataset to a more complaint format (snakecase).
  - Remove columns that have no relevant information.
  - Parse dates to datetime format and some string represented columns to numeric values.
  - Identify and remove outliers.
  - Identify data that could to be translated from German to English. 
- Answering questions like:
  - Which car brands are more popular?
  - What is the mean mileage as well as the mean price for the top used cars brands?
  - Which are the most common brand/model combinations?
  - Search for patterns in the mileage.
  - Compare damaged and non-damaged cars from a selling point perspective. 
  
