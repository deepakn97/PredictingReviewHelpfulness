# Using sqlite3
# Create Amazon_data.db in data folder

# ----------------------- use funtion getDatatoCSV_sql to create csv file

# ----------------- use the below lines to import csv in sql database and create product_data table
# .mode csv
# .import reviews_Amazon_Instant_Video_5.csv  product_data
# .exit

# %%
import source.utility as util
import sqlite3 as sql
import os

# %%
# Function to create CSV file to be imported in sql
util.getDatatoCSV_sql(os.path.abspath('./data/reviews_Amazon_Instant_Video_5.json.gz'), 'Amazon_Instant_Video')

# %%
def create_user_table():
    db = sql.connect(os.path.abspath('./data/Amazon_data.db'))
    cur = db.cursor()
    cur.execute("SELECT count(*) from sqlite_master where type = 'table' and name = 'user_data';")
    flag = cur.fetchall()[0][0];
    if flag == 0 :
        cur.execute("CREATE TABLE if not exists user_data (userId TEXT PRIMARY KEY, X1 FLOAT, X2 FLOAT, UR FLOAT);")
        cur.execute("INSERT INTO user_data(userId,X1,X2,UR) SELECT DISTINCT(reviewerID),0,0,0 FROM product_data order by reviewerID;")
        db.commit()
        print ("User_data table was created with entries")
    else :
        print ("user_data table already exists")
    db.close()

def Drop_user_table():
    db = sql.connect(os.path.abspath('./data/Amazon_data.db'))
    cur = db.cursor()
    cur.execute("DROP TABLE if exists user_data ;")
    db.close()

def update(): # Need to update this function to update the data table
    db = sql.connect(os.path.abspath('./data/Amazon_data.db'))
    cur = db.cursor()
    cur.execute("SELECT * from product_data order by reviewerID, reviewTime;")
    rows = cur.fetchall();
    print (rows[:10]);


# %%
create_user_table()

# %%
update()
|
