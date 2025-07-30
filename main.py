if __name__ == "__main__":
    from src.models.train_model import training
    from src.utils.metrics import results

    pipeline, X, Y, x_test, y_test = training()
    df_cv = results(pipeline, x_test, y_test)
