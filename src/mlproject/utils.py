import os
import sys
import mysql.connector
import pymysql
import pandas as pd
from dotenv import load_dotenv
from src.mlproject.logger import logger
from src.mlproject.exception import CustomException

# Load environment variables
load_dotenv()
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
database = os.getenv("db")  # ✅ Fixed variable name

def read_sql_data():
    logger.info("Reading from MySQL")
    try:
        # Connect to MySQL database
        mydb = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        logger.info("Connection established successfully!")

        # Read data using pandas
        query = "SELECT * FROM student"
        df = pd.read_sql(query, mydb)  # ✅ MySQL Connector supports pandas directly

        logger.info("Data fetched successfully!")
        print(df.head(5))

        return df
    # if you want to use pymysql
    # try:
    #     mydb = pymysql.connect(
    #         host=host,
    #         user=user,
    #         password=password,
    #         database=database
    #     )
    #     logger.info(f"Connection established: {mydb}")

    #     # Use cursor for SQL execution
    #     with mydb.cursor() as cursor:
    #         cursor.execute("SELECT * FROM student")
    #         data = cursor.fetchall()
    #         df = pd.DataFrame(data, columns=[col[0] for col in cursor.description]) 
        
    #     logger.info("SQL Query executed successfully")
    #     print(df.head(5))
    #     return df

    except Exception as e:
        logger.error(f"Error in read_sql_data: {str(e)}")
        raise CustomException(e, sys)

