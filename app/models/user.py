from ..extensions import db

def create_user(user):
    return db.users.insert_one(user)

def get_user(username):
    return db.users.find_one({'username': username})
