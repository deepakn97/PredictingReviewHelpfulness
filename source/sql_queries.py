# .mode csv
# .import file.csv  table_name
# .exit


# select product_id,helpful,reviewerId,reviewTime from product_data  order by reviewerid ,reviewtime limit 1000;
# select reviewerid, count(*) as count from product_data group by reviewerid order by count;  { minimum reviews by each user : 5 }
# create table user_table(userId TEXT PRIMARY KEY,helpful FLOAT ,nothelpful FLOAT);
#
# insert into user_table(userid,helpful,nothelpful)
# select distinct(reviewerID),0,0
# from product_data
# order by reviewerID;

# %%
import source.utility as util
import sqlite3
import os

# %%
# Function to create CSV file to be imported in sql
util.getDatatoCSV_sql(os.path.abspath('../PredictingReviewHelpfulness/data/reviews_Amazon_Instant_Video_5.json.gz'), 'Amazon_Instant_Video')

conn = 
