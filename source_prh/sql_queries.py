# Using sqlite3
# Create Amazon_data.db in data folder

# ----------------------- use funtion getDatatoCSV_sql to create csv file

# ----------------- use the below lines to import csv in sql database and create product_data table
# .mode csv
# .import reviews_Amazon_Instant_Video_5.csv  product_data
# .exit

# %%
import source_prh.utility as util
import sqlite3 as sql
import os
import csv
import pandas as pd
# %%
# Function to create CSV file to be imported in sql
util.getDatatoCSV_sql(os.path.abspath('./data/reviews_Amazon_Instant_Video_5.json.gz'), 'Amazon_Instant_Video')

# %%
def create_product_data_table():
    db = sql.connect(os.path.abspath('./data/Amazon_data.db'))
    cur = db.cursor()
    cur.execute("SELECT count(*) from sqlite_master where type = 'table' and name = 'product_data';")
    flag = cur.fetchall()[0][0];

    if flag == 0 :
        cur.execute("CREATE TABLE if not exists product_data (slno INT PRIMARY KEY, product_id TEXT, reviewText TEXT, summary TEXT,  reviewTime INT, overall FLOAT,  reviewerID TEXT, review_rating FLOAT, ur FLOAT);")
        df = pd.read_csv('./data/reviews_Amazon_Instant_Video_5.csv')
        df = df.rename(columns={"Unnamed: 0" : "slno"})
        df = df[['slno','product_id','reviewText','summary','reviewTime','overall','reviewerID','review_rating','ur']]

        for id,rows in df.iterrows():
            cur.execute("INSERT INTO product_data(slno, product_id, reviewText, summary, reviewtime, overall, reviewerID, review_rating, ur) values(?,?,?,?,?,?,?,?,?);",rows)
        db.commit()
        print ("product_data table was created with entries")

    else :
        print ("product_data table already exists")
    db.close()

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

def Drop_table(t_name):
    db = sql.connect(os.path.abspath('./data/Amazon_data.db'))
    cur = db.cursor()
    cur.execute("DROP TABLE if exists "+t_name +";")
    db.close()

def update(): # Need to update this function to update the data table
    db = sql.connect(os.path.abspath('./data/Amazon_data.db'))
    cur = db.cursor()
    cur.execute("SELECT slno,reviewerID,review_rating from product_data order by reviewerID, reviewTime;")
    rows = cur.fetchall();
    for slno,uid, rr in rows :
        if(rr<0.5):
            cur.execute("update user_data set x1=x1, x2 = x2+1.0  where userId = ?;",(uid,))
        elif(rr>0.5):
            cur.execute("update user_data set x1=x1+1.0, x2 = x2 where userId = ?;",(uid,))
        cur.execute("update user_data set ur=(x1-x2)/(x1+x2) where userId = ?;",(uid,))
        cur.execute("update product_data set ur = (select ur from user_data where userid = ?) where slno = ?;",(uid,slno,))
    db.commit()
    db.close()
# %%
create_product_data_table()
create_user_table()

# %%
Drop_table('product_data')
Drop_table('user_data')
# %%
update()
# %%
