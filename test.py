from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(prefix="ann_weighted_position2", folds=10, algorithms=["ann_weighted_position"])
    c.process()