from utilities import load_csv2df
from linear_regression import LinearRegression

my_df = load_csv2df('data.csv', 'CO2 Emissions(g/km)')
print(my_df)
n = LinearRegression(my_df)

try:
    n.train(15)
except KeyboardInterrupt:
    print('Saving....')
    n.save_model('model12')