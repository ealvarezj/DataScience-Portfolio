## 2. Exploring Ebay Kleinanzeigen car sales data.

In this project I explored the `Ebay Car Sales Dataset` from the [Ebay Kleinanzeigen](https://www.ebay-kleinanzeigen.de/) German website, a division for classified advertisements' website with sections devoted to jobs, housing, services, community service, gigs, and cars. Similar to "Craiglist" in the United States. The dataset was crawled from the website directly. 

![kleinanzeigen](/DataScience-Portfolio/images/kleinanzeigen.png)

The analysis was done completely using python, specially using the `pandas` library for data manipulation. The goals of this project where:

- Data cleaning:
  - rename features in the dataset to a more complaint format (snakecase).
  - Remove columns that have no relevant information.
  - Parse dates to datetime format and some string represented columns to numeric values.
  - Identify and remove outliers.
  - Identify data that could to be translated from German to English. 
- Answering questions like:
  - Which car brands are more popular?
  - What is the mean mileage as well as the mean price for the top used cars brands?
  - Which are the most common brand/model combinations?
  - Search for pattern in the mileage
  - Compare damaged and non-damaged cars from a selling point perspective. 
  
