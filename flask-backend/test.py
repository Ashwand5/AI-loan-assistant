from pymongo import MongoClient

uri = "mongodb+srv://ashwandashwands:tLyPh5XHQXR7Y0J1@medical-bot.e6pwt.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)

try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB!")
except Exception as e:
    print("❌ Connection failed:", e)
