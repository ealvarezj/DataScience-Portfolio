## Using SQL to answer business questions


<div align="center">

  **Run this project using:** [STATIC PREVIEW](https://nbviewer.jupyter.org/github/ealvarezj/Data-Science-Portfolio/blob/main/SQL_Projects/SQL_Music_Store/SQL_Music_Store_Project.ipynb) or [![sql](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ealvarezj/Data-Science-Portfolio/main?filepath=SQL_Projects/SQL_Music_Store/SQL_Music_Store_Project.ipynb)

</div>

In this project I analyzed data from a music store saved as an SQLite database  Here is an image of the database schema. 



<!-- <details><summary>CLICK ME</summary>
<p> -->

![db_structure](/DataScience-Portfolio/images/chinook-schema.svg)

<!-- </p>
</details> -->

The analysis was done completely using only SQL and some python code for creating database connection functions and visualizations. Some goals of this project where:

- Querying data from SQlite and parsing the output using `pandas` to a `pandas.DataFrame`.
- Use common SQL methods like `INNER, LEFT JOINS` `Aggregation functions` and `Subqueries`
- Use more advanced SQL methods like `common table expressions (CTE)`, `VIEWS`, `CASE  statements` as well as `UNION` and `EXCEPT` operations. 

- Answering questions like:
  - What is the most selling genre in the USA?, and using the results to make decisions about new label contracts.
  - Compare sales agents' performance.
  - Analyze sales by country.
  - Inspecting the behavior and preferences of customers regarding complete album purchase orders or mixed track orders.
  - Searching for the most common artist for each playlist
  - Finding out how much of the tracks in the store have been bought at least one time, and which have not been bought yet.
  - Look at popularity and revenue based on music audio files format.


<!-- Binder latest URl

https://mybinder.org/v2/gh/ealvarezj/Data-Science-Portfolio/main?filepath=SQL_Projects/SQL_Music_Store/SQL_Music_Store_Project.ipynb


put the path to the file using the main branch

 -->