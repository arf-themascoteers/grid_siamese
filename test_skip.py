from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(prefix="skip", folds=10, algorithms=[
        "ann_avg_skip"
    ])
    c.process()