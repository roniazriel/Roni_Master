import torch
from sklearn.model_selection import train_test_split
import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetRegressor

''' 1. TabNet'''
''' Modeling '''
''' split to train, test and validation sets '''
n_targets = 2

train_df = grouped_data.sample(frac=0.8,random_state=200) #random state is a seed value
test_df = grouped_data.drop(train_df.index)

X = train_df.drop(columns=['Manipulability_Rates','Success_Rates']).values
Y = train_df[['Manipulability_Rates','Success_Rates']].values
X_train, x_valid, y_train, y_valid = train_test_split(train_df.drop(columns=['Manipulability_Rates','Success_Rates'] ), train_df[['Manipulability_Rates','Success_Rates']], test_size=0.3)
x_test = test_df.drop(columns=['Manipulability_Rates','Success_Rates'] )
y_test = test_df[['Manipulability_Rates','Success_Rates']]

x_train =X_train.to_numpy()
y_train=y_train.to_numpy()
x_valid =x_valid.to_numpy()
y_valid =y_valid.to_numpy()
x_test =x_test.to_numpy()
y_test =y_test.to_numpy()


print("x train\n", x_train.shape)
print("y train\n", y_train.shape)

print("x valid\n", x_valid.shape)
print("y valid\n", y_valid.shape)

print("x test\n", x_test.shape)
print("y test\n", y_test.shape)

clf1= TabNetRegressor()
# fit the model
clf1.fit(
    x_train,y_train,
    eval_set=[(x_train, y_train), (x_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['mae','mse'],
    max_epochs=1000 , patience=50,
    batch_size=256, virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

preds = clf1.predict(x_test)

test_mse = mean_squared_error(y_pred=preds, y_true=y_test)
test_mae = mean_absolute_error(y_test, preds)

print(f"BEST VALID SCORE: {clf1.best_cost}")
print(f"MSE TEST SCORE: {test_mse}")
print(f"MAE TEST SCORE: {test_mae}")

def Objective(trial):
    mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
    n_da = trial.suggest_int("n_da", 56, 64, step=4)
    n_steps = trial.suggest_int("n_steps", 1, 3, step=1)
    gamma = trial.suggest_float("gamma", 1., 1.4, step=0.2)
    n_shared = trial.suggest_int("n_shared", 1, 3)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
    tabnet_params = dict(n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
                     lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                     mask_type=mask_type, n_shared=n_shared,
                     scheduler_params=dict(mode="min",
                                           patience=trial.suggest_int("patienceScheduler",low=3,high=10), # changing sheduler patience to be lower than early stopping patience
                                           min_lr=1e-5,
                                           factor=0.5,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=0,
                     ) #early stopping
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    CV_score_array    =[]
    for train_index, test_index in kf.split(X):
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = Y[train_index], Y[test_index]
        regressor = TabNetRegressor(**tabnet_params)
        regressor.fit(X_train=X_train, y_train=y_train,
                  eval_set=[(X_valid, y_valid)],
                  patience=trial.suggest_int("patience",low=15,high=30), max_epochs=trial.suggest_int('epochs', 1, 100),
                  eval_metric=['mse','mae'])
        CV_score_array.append(regressor.best_cost)
    avg = np.mean(CV_score_array)
    return avg

study = optuna.create_study(direction="minimize", study_name='TabNet optimization')
study.optimize(Objective, timeout=6*60) #5 hours
TabNet_params = study.best_params

final_params = dict(n_d=TabNet_params['n_da'], n_a=TabNet_params['n_da'], n_steps=TabNet_params['n_steps'], gamma=TabNet_params['gamma'],
                     lambda_sparse=TabNet_params['lambda_sparse'], optimizer_fn=torch.optim.Adam,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                     mask_type=TabNet_params['mask_type'], n_shared=TabNet_params['n_shared'],
                     scheduler_params=dict(mode="min",
                                           patience=TabNet_params['patienceScheduler'],
                                           min_lr=1e-5,
                                           factor=0.5,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=0,
                     )
epochs = TabNet_params['epochs']

regressor = TabNetRegressor(**final_params)
regressor.fit(x_train,y_train,
    eval_set=[(x_train, y_train), (x_valid, y_valid)],
    eval_name=['train', 'valid'],
    patience=TabNet_params['patience'], max_epochs=epochs,
    eval_metric=['mse','mae'])