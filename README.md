## Predicting Review Helpfulness
#### Group_5:
* **Deepak Nathani (ME15BTECH11009)**
* **Yash Pitroda (ES15BTECH11020)**
* **Sahil Yerawar(CS15BTECH11044)**

#### Project description
We did this project as a part of the course Information Retrieval here at IIT Hyderabad.</br>
There are a lot of eCommerce websites with a lot of reviews for each product on these websites. Most of the websites provide an option of marking each review helpful or not helpful, to help the future customers interested in buying that commodity. However, most of the reviews remain unvoted and are of no use. But what if we could predict the helpfulness given the previous review data and user data. This thought motivated us to work on this project.</br>



#### File descriptions:
* *model.py* - File has code for model definitions and model training.
* *sql_queries.py* - Contains code for creating database and getting user ratings
* *utility.py* - Contains utility codes for pre-processing and
* *keras35.yml* - Conda environment file to create a replica the environment used for testing and training



#### Instructions to run code:
1. Install **miniconda** from [here](https://conda.io/miniconda.html) for creating virtual environment.  

2. Use keras35.yml file to set up the environment  using the following command:</br>`$ conda env create -f environment.yml`

3. Activate conda environment with following:
</br>`$ source activate keras35`
4. Add the project folder in PYTHONPATH with following command:</br>
`$ conda develop .`

5. Run the code using `python ./source_prh/model.py`
