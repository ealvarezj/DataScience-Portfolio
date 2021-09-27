## Exploring NYC Open Data and Schools.



 <div align="center">

  **Run this project using:** [STATIC PREVIEW](https://nbviewer.jupyter.org/github/ealvarezj/Data-Science-Portfolio/blob/main/Guided_Project_Schools/Schools.ipynb) or [![schools](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ealvarezj/binder_env/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fealvarezj%252FData-Science-Portfolio%26urlpath%3Dtree%252FData-Science-Portfolio%252FGuided_Project_Schools%252FSchools.ipynb%26branch%3Dmain)

</div>


![](https://opendata.cityofnewyork.us/wp-content/themes/opendata-wp/assets/img/nyc-open-data-logo.svg)



In this project I analyzed New York City public schools data. Data about school grades, class sizes, dropout and graduation rates is available in the NYC OpenData website. The objective of this project is to joined multiple dataset containing different type of data about schools and find possible correlations between exam results and demographics. The data used consist of the following datasets:

- [SAT scores by school](https://data.cityofnewyork.us/Education/SAT-Results/f9bf-2cp4) - SAT scores for each high school in New York City
- [School attendance](https://data.cityofnewyork.us/Education/School-Attendance-and-Enrollment-Statistics-by-Dis/7z8d-msnt) - Attendance information for each school in New York City
- [Class size](https://data.cityofnewyork.us/Education/2010-2011-Class-Size-School-level-detail/urz7-pzb3) - Information on class size for each school
- [AP test results](https://data.cityofnewyork.us/Education/AP-College-Board-2010-School-Level-Results/itfs-ms3e) - Advanced Placement (AP) exam results for each high school (passing an optional AP exam in a particular subject can earn a student college credit in that subject)
- [Graduation outcomes](https://data.cityofnewyork.us/Education/Graduation-Outcomes-Classes-Of-2005-2010-School-Le/vh2h-md7a) - The percentage of students who graduated and other outcome information
- [Demographics](https://data.cityofnewyork.us/Education/School-Demographics-and-Accountability-Snapshot-20/ihfw-zy9j) - Demographic information for each school
- [School survey](https://data.cityofnewyork.us/Education/NYC-School-Survey-2011/mnz3-dyi8) - Surveys of parents, teachers, and students at each school

The SAT, or Scholastic Aptitude Test, is a test that high school seniors in the U.S. take every year. The SAT has three sections, each is worth 800 points. Colleges use the SAT to determine which students to admit. High average SAT scores are usually indicative of a good school.

Due to the large diversity of races in NYC, the main aim is to find relationships between race and performance in schools. We would also like to see later on if wealth plays a role in school performance by adding data about wealth status in the different school districts in the city. First lets start by cleaning and reading the data. 
  
Some questions we would like to answer are:

- Is there a correlation between class size and SAT scores?
- Which neighborhoods have the best schools?
- Is there a relationship between socio-economical status and performance on SAT scores?, Are students going to schools in more wealthy neighborhoods more successful.
- Investigate the differences between parent, teacher, and student responses to surveys.
