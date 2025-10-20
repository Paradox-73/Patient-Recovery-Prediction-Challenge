
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

def get_models():
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
        'AdaBoost Regressor': AdaBoostRegressor(random_state=42),
        'K-Neighbors Regressor': KNeighborsRegressor(),
        'Lasso': Lasso(random_state=42),
        'Ridge': Ridge(random_state=42)
    }
    return models
