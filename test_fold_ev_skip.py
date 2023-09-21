from fold_evaluator import FoldEvaluator

if __name__ == "__main__":
    c = FoldEvaluator(prefix="skip", folds=10, algorithms=["ann_skip"])
    c.process()