# Master Thesis (Data Science Project)

This project contains procedures regarding my master thesis: **"Design and implementation of a machine learning model for the prediction of production times, as an optimization tool for proposal management tasks for die forged products."**. This has been my most comprenhensive Data Science Project until now. Due to a nondisclosure agreement, I am not able to share any data or code regarding the project. For this reason I am only going to provide information about the procedure used in the project.

---
## Introduction 

Already in the development phase of new high-quality components, engineers are faced with the question of the relationship between function and manufacturing costs. Often, the intended use case and the resulting geometric design of a component determine which manufacturing process might make sense from a production and commercial point of view. The decision-making processes can take a long time, which unnecessarily prolongs the development time of products. One of the main aims in the product development department is to provide customers' development engineers with the most accurate possible cost, requirement, and quality-related advantages of the closed-die forging process to help facilitate quick decision-making. 
A high degree of technical and experiential knowledge is required to meet this standard, which is already applied during the pre-calculation phase through the development of technically sophisticated workflow sequences. It quickly becomes clear what an immense effort hides behind the processing of a customer product request. Information from the most diverse production areas must be collected repeatedly and evaluated and classified concerning its relevance for the best possible manufacturing process. 

This work tries to tackle, employing `Supervised Machine Learning`, the proposal management process in product development. It should be possible to noticeably accelerate the processing time of the pre-calculation process with machine learning methods and at the same time maintain a high quality of the proposal management results. This use case aims to reduce the dependency on Empirical Research for assignment of production times, particularly for the forging process of closed die forged parts, eliminating the need for FEM Simulations to back up first-time proposals.

`Closed die forging` belongs according to the German Institute for Standardisation DIN to the pressing forming processes besides indentation, rolling and open die forging. For the forging process, forging dies with the negative geometries are used. A preheated ingot is put between the dies, force is applied (e.g. forging hammer), the material starts to flow inside the die until the end geometry is filled. 


<table>
<thead>
<tr>
    <th>Hydraulic Counterblow Hammer</th>
    <th>Closed die forging process</th>
</tr>
</thead>
<tbody>
<tr>
    <td><img src="https://www.dango-dienenthal.de/fileadmin/user_upload/applications/close-die-forging/DD-Gesenkschmiede_1.jpg" alt="hammer" width="400"/></td>
    <td><img src="https://slideplayer.org/slide/1343508/3/images/12/Gesenkschmieden-Prinzipdarstellung%3A.jpg" alt="forming_process" width="400"/></td>
</tr>
</tbody>
</table>



Due to the way materials and especially steel aloys behave in the plasticity state (this is the state in which material deforms and can not return to its initial state) it is not possible using a generalized formula to calculate the required time for the forging process. The plastic deformation behavior from materials cannot be linearly explained. Depending on temperature, chemical composition of the material at varying strain rates, the amount of force needed to achieve the desired true strain varies. Below is a Comparison of flow curves for a C15 steel, with different temperatures and strain
rates.  


{% include image.html url="2021-09-29-14-26-06.png" description="Comparison of flow curves for a C15 steel, with different temperatures and strain
rates." width="500" %}


Until now, the prediction of the materials' behaviour on the parts' geometry can only be achieved using FEM (Finite element method) simulations. The idea of this project is to find a faster way of achieving a prediction of the forging time for which the deformation force plays an important role. 


## Data availability

Data availability has become nowadays a topic of high relevance for most companies. Technologies like Data Mining are used to find interesting patterns in data that might well already been there, stored and archived or being generated on a daily basis. The learning dataset is composed of data coming from different sources. All parameters where collected for 313 different products, for which all 19 features where available. The resulting dataset is a mixture of continous and categorical variables. This dataset was used as the master dataset. Later on records from the feedback system where joined with the master dataset. 



{% include image.html url="2021-09-29-16-17-01.png" description="Data flow framework" width="500" %}



## Exploratory Data Analysis (EDA)

Data analysis is a crucial step in any Machine Learning problem, is a way to introduce and understand data before moving further to the modeling process. Data analysis provides the analyst with important information about data types, relation between variables in a dataset, their shape, form, distribution as well as descriptive statistics. In data analysis patterns in the data can be discovered by means of numerical tests and visualization techniques. This is in fact one of the most time-consuming steps on any project, since it requires data to be consolidated and prepared for the analysis.



<table>
<thead>
  <tr>
    <th>Correlation Matrix</th>
    <th>Data distribution</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><img src="2021-09-29-16-33-39.png" alt="matrix" width="400"/></td>
    <td rowspan="3"><img src="raincloud.png" alt="raincloud" width="600"/></td>
  </tr>
  <tr>
    <th>Analysis of residuals</th>
  </tr>
  <tr>
    <td><img src="residuals.png" alt="residuals" width="400"/></td>
  </tr>
</tbody>
</table>


## Model evaluation
There are numerous regression algorithms already implemented in libraries like `scikit-learn` that one can choose from. I chose a simple linear regression models as my baseline model, since it is fast to train due to its vectorized closed form implementation, and easy to interpret. After getting first results and model accurary a set of different models where tested on the same dataset to create a benchmark. 


| Model                           | MAE     | MSE      | RMSE    | R2      | RMSLE  | MAPE   | TT (Sec) |
|---------------------------------|---------|----------|---------|---------|--------|--------|----------|
| CatBoost Regressor              | 4.2354  | 50.8014  | 6.9793  | 0.8645  | 0.2672 | 0.2042 | 1.618    |
| Extra Trees Regressor           | 4.3761  | 56.3788  | 7.2692  | 0.8534  | 0.2778 | 0.2097 | 0.029    |
| Random Forest Regressor         | 4.4652  | 58.7568  | 7.4328  | 0.8481  | 0.2661 | 0.2050 | 0.038    |
| Gradient Boosting Regressor     | 4.5829  | 60.1312  | 7.5515  | 0.8387  | 0.2703 | 0.2183 | 0.014    |
| Light Gradient Boosting Machine | 4.5967  | 66.1371  | 7.8272  | 0.8363  | 0.2817 | 0.2099 | 0.008    |
| Extreme Gradient Boosting       | 4.5624  | 67.4343  | 7.9526  | 0.8192  | 0.2789 | 0.2110 | 12.492   |
| AdaBoost Regressor              | 6.6773  | 81.7521  | 8.9387  | 0.7735  | 0.3999 | 0.4283 | 0.012    |
| Decision Tree Regressor         | 5.2150  | 87.8739  | 9.1760  | 0.7619  | 0.3138 | 0.2339 | 0.004    |
| K Neighbors Regressor           | 5.3629  | 88.5753  | 9.2931  | 0.7544  | 0.3071 | 0.2343 | 0.005    |
| Ridge Regression                | 6.6509  | 95.4289  | 9.5335  | 0.7442  | 0.3869 | 0.3239 | 0.004    |
| Least Angle Regression          | 6.6597  | 95.6338  | 9.5440  | 0.7433  | 0.3877 | 0.3246 | 0.005    |
| Linear Regression               | 6.6597  | 95.6337  | 9.5440  | 0.7433  | 0.3877 | 0.3246 | 0.004    |
| Elastic Net                     | 6.9408  | 98.6090  | 9.7182  | 0.7374  | 0.3786 | 0.3313 | 0.004    |
| Lasso Regression                | 6.9496  | 98.9866  | 9.7357  | 0.7366  | 0.3792 | 0.3319 | 0.004    |
| Huber Regressor                 | 6.5085  | 100.9395 | 9.7533  | 0.7321  | 0.3367 | 0.3023 | 0.006    |
| Bayesian Ridge                  | 7.0413  | 100.8291 | 9.8349  | 0.7312  | 0.3786 | 0.3334 | 0.004    |
| Orthogonal Matching Pursuit     | 8.0436  | 146.5258 | 11.7492 | 0.6310  | 0.3851 | 0.3827 | 0.004    |
| Passive Aggressive Regressor    | 9.8682  | 207.8557 | 13.9189 | 0.4282  | 0.5120 | 0.3922 | 0.004    |
| Lasso Least Angle Regression    | 15.6397 | 406.6446 | 19.7724 | -0.0291 | 0.7431 | 1.0209 | 0.004    |


As we can see, benchmarking different models with their standard configurations gives us a pretty good start in terms of accuracy. Accuracy was measured using the **RMSE** (Root Mean Squared Error) metric. These models have still the potential to be optimized. Model optimization goes beyond numerical optimization e.g. **Grid Search** or **Random Search**. Performance of a model can also be improved by carefully performing **feature engineering** and **ensenmbling or stacking** models together. In the table above is possible to see, that relatively simple algorithms like **Random Forest** perform better than more complex and modern approaches like **LGBM (Light Gradient Boosting Machine)**. In fact, the difference in performance between the Random forest Regressor and the CatBoost Regressor is marginal in terms of the RMSE (measured in minutes) one small difference is the training time required for CatBoost is significantly higher, which for the final size of the dataset does not make a lot of difference but if scaled could potentially present an issue.

### Hyperparameter Optimization and Dimensionality reduction

Since Random Forest is supported natively by **scikit-learn** we can make use of the **GridSearchCV** and **RandomSearhCV** classes to perform Hyperparameter Optimization. The following table show the optimization proocess for the Random Forest Regressor estimator. $$ 4=3 $$


| Hyperparameter                                   | Value range                                                                             | GSearch        | RSearch        |
|--------------------------------------------------|-----------------------------------------------------------------------------------------|----------------|----------------|
| Max. depth                                       | $$ None, [2,10] \in \mathbb{Z} $$                                                       | 8           | None           |
| Max. features                                    | $$ {auto, sqrt, log2} $$                                                          | sqrt           | log2           |
| Max. leaf nodes                                  | $$ {None, [2,10] \in \mathbb{Z} $$                                                       | None           | None           |
| Criterion                                        | $$ mse, friedeman mse, mae, poisson $$                                                | friedman mse   | mae            |
| ccp alpha                                        | $$ [0, 0.9] \in \mathbb{R} $$                                                               | 0.1          | 0           |
|**Metric**                 | **Resources**                                                                              |                |                |
|$$ R^{2} $$ with $$ CV = 10 $$       | Performed on a machine  with 12 CPU cores at 4.3 Ghz and  16 Gb of RAM | 0.8562         | 0.8574         |
|$$ RMSE $$ with $$ CV = 10 $$        |                                                                                         | 8.089          | 8.104          |
|Number of fits with $$ CV = 5 $$ |                                                                                         | 48600          | 2500           |
|Compute time (seconds)  |                                                                                         | 768            | 40.3           |
|**Baseline model**         | $$ \mathbf{RMSE} $$                                                                         | $$ \mathbf{R^2} $$ | $$ \mathbf{MAE} $$ |
| Random Forest regressor                          | $$ 8.4708 $$                                                                                | $$ 0.8428 $$       | $$ 71.7549 $$      |


