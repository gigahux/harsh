import os

def read_reviews(file_path):
    with open(file_path, 'r') as f:
        reviews = f.readlines()
    return [review.strip() for review in reviews if review.strip()]

def read_prompt(file_path, review):
    with open(file_path, 'r') as f:
        prompt = f.read()
    return prompt.format(review=review)