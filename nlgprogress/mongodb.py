import pymongo
import logging

from app.config import const

# Connection pool
client = pymongo.MongoClient(
    f"mongodb://{const.MONGODB_ADMINUSERNAME}:{const.MONGODB_ADMINPASSWORD}@localhost:27017/")
db = client["nycuQA"]

def mongodb_login(table):
    collection = db[table]
    return collection

def query_mongodb_question(question):
    table = "question"
    collection = mongodb_login(table)

    try:
        # Add appropriate indexes as per your requirements
        collection.create_index("question")

        query_result = collection.find_one({
            "question": {"$regex": question}
        })

        return query_result
    except pymongo.errors.PyMongoError as e:
        logging.error(f"An error occurred while querying MongoDB: {e}")
