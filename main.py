import os
from src.utils import read_reviews, read_prompt
from src.classifier import Classifier

def main():
    # Initialize Classifier
    classifier = Classifier()

    # Read reviews
    reviews = read_reviews(os.path.join('data', 'reviews.txt'))

    # Zero-Shot
    zero_shot_prompt = read_prompt(os.path.join('prompts', 'zero_shot_prompt.txt'), "{review}")
    zero_shot_chain = classifier.classify_zero_shot(zero_shot_prompt)
    print("=== Zero-Shot Classification ===")
    for review in reviews:
        classification = zero_shot_chain.run(review=review)
        print(f"Review: {review}\\nClassification: {classification.strip()}\\n")

    # One-Shot
    one_shot_prompt = read_prompt(os.path.join('prompts', 'one_shot_prompt.txt'), "{review}")
    one_shot_chain = classifier.classify_one_shot(one_shot_prompt)
    print("=== One-Shot Classification ===")
    for review in reviews:
        classification = one_shot_chain.run(review=review)
        print(f"Review: {review}\\nClassification: {classification.strip()}\\n")

    # Few-Shot
    few_shot_prompt = read_prompt(os.path.join('prompts', 'few_shot_prompt.txt'), "{review}")
    few_shot_chain = classifier.classify_few_shot(few_shot_prompt)
    print("=== Few-Shot Classification ===")
    for review in reviews:
        classification = few_shot_chain.run(review=review)
        print(f"Review: {review}\\nClassification: {classification.strip()}\\n")

if __name__ == "__main__":
    main()