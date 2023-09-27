from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(prefix="ann_weighted2_2", folds=10, algorithms=["ann_weighted2"])
    c.process()