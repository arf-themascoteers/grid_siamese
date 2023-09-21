from fold_evaluator import FoldEvaluator

if __name__ == "__main__":
    c = FoldEvaluator(prefix="shared", folds=10, algorithms=["ann_shared"])
    c.process()