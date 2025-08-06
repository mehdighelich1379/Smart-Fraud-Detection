from src.models.train_model import training
from src.utils.metrics import results, cross_val

pipeline, X, Y, x_test, y_test = training()
df_cv = results(pipeline, x_test, y_test)
print(X)
print(df_cv)