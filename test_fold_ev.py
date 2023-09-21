from fold_evaluator import FoldEvaluator

if __name__ == "__main__":
    c = FoldEvaluator(prefix="ann2", folds=10, algorithms=["ann"])
    c.process()