from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(prefix="position", folds=10, algorithms=[
        "ann_centre_only","ann_top_left_only","ann_avg","ann_centric_avg","ann_learnable_avg"
    ])
    c.process()