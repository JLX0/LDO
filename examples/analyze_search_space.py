import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
   x = trial.suggest_float("x", -5, 5)
   y = trial.suggest_float("y", -5, 5)
   z = trial.suggest_categorical("z", ["Tom", "Jack", "Alice"])
   a = trial.suggest_int("a", 1, 10)


   return x**2 + y**2  - a + len(z)


load_local=True

if load_local:
   mod = optunahub.load_local_module("home/j/experiments/LDO/ldo/test_package")
   study = optuna.create_study(sampler=mod.test_sampler(debug=True, seed=42))
   study.optimize(objective, n_trials=100)
else:
   module = optunahub.load_module("samplers/differential_evolution")
   DESampler = module.DESampler
   sampler = DESampler(population_size="auto", F=0.8, CR=0.9, seed=42)

   # Create and optimize the study
   study = optuna.create_study(direction="minimize", sampler=sampler)
   study.optimize(objective, n_trials=100)

print(study.best_trial.value, study.best_trial.params)



