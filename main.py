import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

nba = pd.read_csv('nba_games.csv')
pd.set_option('display.max_columns', None)
# Subset Data to 2010 Season, 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

print(nba_2010.head())
# print(nba_2014.head())

#calculating points of knicks and nets

knicks_pts = nba_2010.pts[nba_2010.fran_id == 'Knicks']

nets_pts = nba_2010.pts[nba_2010.fran_id == "Nets"]

# finding mean differance
diff_means_2010 = (np.mean(knicks_pts))- (np.mean(nets_pts))
print(diff_means_2010)



plot histogram to visualize 
plt.hist(knicks_pts, alpha=0.8, density = True, label='knicks_2010')
plt.hist(nets_pts, alpha=0.8, density = True, label='nets 2010')
plt.legend()
plt.show()

#calculating 2014 points
knicks_pts_2014 = nba_2014.pts[nba_2014.fran_id == 'Knicks']

nets_pts_2014 = nba_2014.pts[nba_2014.fran_id == "Nets"]
# calculate the mean diff

diff_means_2014 = (np.mean(knicks_pts_2014))- (np.mean(nets_pts_2014))
print(diff_means_2014)


#plotting histogram to 

plt.hist(knicks_pts_2014, alpha=0.8, density = True, label='knicks 2014')
plt.hist(nets_pts_2014, alpha=0.8, density = True, label='nets 2014')
plt.legend()
plt.show()



# Visualizing the points scored by Each Franchise
# sns.boxplot(x = 'fran_id', y = "pts", data = nba_2010)
# plt.show()

# find if the teams are winning when playin At home and losing at away with frequency table
#calculating freq
location_result_freq = pd.crosstab(nba_2010.game_result, nba_2010.game_location)
print(location_result_freq)

#calculate the location_result_proportions
location_result_proportions = location_result_freq/ len(nba_2010)
print(location_result_proportions)


#elevating similarity between two variables with CHI SQUARE TEST

chi2, pval, df, exp = chi2_contingency(location_result_freq)
print(chi2, exp)


#calculating covariance between point diff and forecast
pointdiff_forecast_cov = np.cov(nba_2010.point_diff, nba_2010.forecast)
print(pointdiff_forecast_cov)

#calculate correlation between forecast and point diff
pointdiff_forecast_corr = pearsonr(nba_2010.point_diff,nba_2010.forecast)
print(pointdiff_forecast_corr)

#generate a scatterplot
plt.scatter(x= 'forecast', y = 'point_diff', data = nba_2010)
plt.xlabel('Forecasted Win Prob.')
plt.ylabel('Point Differential')
plt.show()

#therefore we conclude that there is correlation between point difference and forecasting



