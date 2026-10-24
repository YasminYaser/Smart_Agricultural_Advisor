"""Train both models and save them to ./models
Run:  python train.py   (the two CSV files can sit in the project root or in ./data)
"""
from model_utils import train_all

if __name__ == "__main__":
    train_all(save=True, verbose=True)
