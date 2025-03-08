import os
import sys
import mysql.connector
import pymysql
import pandas as pd
import requests
from dotenv import load_dotenv
from src.mlproject.logger import logger
from src.mlproject.exception import CustomException

import pickle
import numpy as np

# Load environment variables
load_dotenv()
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
database = os.getenv("db")  # ✅ Ensure correct env variable name

# 🔹 **1️⃣ Read from MySQL Database (Default)**
def read_mysql_data():
    """
    Reads data from MySQL Database using mysql.connector
    """
    logger.info("Reading from MySQL")
    try:
        mydb = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        logger.info("Connection established successfully!")

        query = "SELECT * FROM student"
        df = pd.read_sql(query, mydb)

        logger.info("Data fetched successfully from MySQL")
        preview_data(df)

        return df
    except Exception as e:
        logger.error(f"Error in read_sql_data: {str(e)}")
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        logger.info(f"Saving object at path: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Object saved at path: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)   






















# 🔹 **2️⃣ Read from SQL Database using  (Default)**
def read_pymysql_data():
    """
    Reads data from MySQL Database using pymysql
    """
    logger.info("Reading from pymysql")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        logger.info("Connection established successfully!")

        # Using cursor for execution
        with mydb.cursor() as cursor:
            cursor.execute("SELECT * FROM student")
            data = cursor.fetchall()
            df = pd.DataFrame(data, columns=[col[0] for col in cursor.description]) 

        logger.info("Data fetched successfully using pymysql")
        print(df.head(5))
        return df
        
    except Exception as e:
        logger.error(f"Error in read_sql_data: {str(e)}")
        raise CustomException(e, sys)


# 🔹 **2️⃣ Read from a CSV File**
def read_csv_data(filepath):
    """
    Reads data from a CSV file
    """
    logger.info(f"Reading CSV file from {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info("CSV Data read successfully")
        preview_data(df)
        return df
    except Exception as e:
        logger.error(f"Error reading CSV: {str(e)}")
        raise CustomException(e, sys)


# 🔹 **3️⃣ Read from an Excel File**
def read_excel_data(filepath, sheet_name=0):
    """
    Reads data from an Excel file
    """
    logger.info(f"Reading Excel file from {filepath}")
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        logger.info("Excel Data read successfully")
        preview_data(df)
        return df
    except Exception as e:
        logger.error(f"Error reading Excel: {str(e)}")
        raise CustomException(e, sys)


# 🔹 **4️⃣ Read from a JSON File**
def read_json_data(filepath):
    """
    Reads data from a JSON file
    """
    logger.info(f"Reading JSON file from {filepath}")
    try:
        df = pd.read_json(filepath)
        logger.info("JSON Data read successfully")
        preview_data(df)
        return df
    except Exception as e:
        logger.error(f"Error reading JSON: {str(e)}")
        raise CustomException(e, sys)


# 🔹 **5️⃣ Read from an API Endpoint**
def read_api_data(api_url):
    """
    Fetches data from an API endpoint and converts it into a DataFrame
    """
    logger.info(f"Fetching data from API: {api_url}")
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        logger.info("API Data fetched successfully")
        preview_data(df)
        return df
    except Exception as e:
        logger.error(f"Error fetching API data: {str(e)}")
        raise CustomException(e, sys)


# 🔹 **6️⃣ Preview Data Function**
def preview_data(df, rows=5):
    """
    Prints the first few rows of a DataFrame
    """
    if df is not None and not df.empty:
        print("\n🔹 **Data Preview:**")
        print(df.head(rows))
        print("\n🔹 **Data Shape:**", df.shape)
    else:
        print("⚠️ Data is empty or not loaded properly!")


"""
-------------------------------------
✅ Usage Examples:
-------------------------------------

🔹 Read from MySQL:
df = read_sql_data()

🔹 Read from CSV:
df = read_csv_data("data.csv")

🔹 Read from Excel:
df = read_excel_data("data.xlsx", sheet_name="Sheet1")

🔹 Read from JSON:
df = read_json_data("data.json")

🔹 Fetch from API:
df = read_api_data("https://api.example.com/data")

"""

