from fold_evaluator import FoldEvaluator

if __name__ == "__main__":
    c = FoldEvaluator(prefix="ann3", folds=10, algorithms=["ann3"])
    c.process()