import numpy as np
import pandas as pd
import data as data
from constants import POPULATION_SCALE, MIGRATION_THRESHOLD, PROCESSES, SPLITS
from gos import Globe
from multiprocessing import Manager

import sys
sys.path.append("optimization/")
from optimization import DE

import sys

w1 = 0
w2 = 0
w3 = 0
w4 = 0
w5 = 0

np.random.seed(1000)

import time
print ('\n'*3, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
f = open('output.txt', 'a')
print ('\n'*3, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file = f)
f.close()

netM = pd.read_csv("netM.csv")
for index, row in netM.iterrows():
    if (row['NetMigration']).find("+AC0-") == -1:
        row['NetMigration'] = int(float(row['NetMigration']) / 1000)
    else:
        row['NetMigration'] = int(float(row['NetMigration'].replace('+AC0-', '')) * (-1) / 1000)
#print(netM)


# The attributes for each agent.
world_columns = ["Country", "Income", "Employed", "Attachment", "Location", "Migration"]


def generate_agents(df, country, population):
    """
    Generate a dataframe of agents for a country where population
    is the number of agents to be created.
    """
    def max_value(attribute):
        return df[attribute].max()
    # Turn this on for truly random output from each process.
    # pid = mp.current_process()._identity[0]
    rand = np.random.mtrand.RandomState(0)
    country_data = df[df.index == country].to_dict("records")[0]
    gdp = country_data["GDP"]
    income_array = gdp / 10 * rand.chisquare(10, population).astype('float32')
    unemployment_rate = float(country_data["Unemployment"] / 100.0)
    employment_array = rand.choice([True, False], population,
                                   p=[1 - unemployment_rate, unemployment_rate])
    attachment_array = (country_data["Fertility"] *
                        rand.triangular(0.0, 0.5, 1.0, population) /
                        max_value("Fertility")).astype('float32')
    frame = pd.DataFrame({
        "Country": pd.Categorical([country] * population, list(df.index)),
        "Income": income_array,
        "Employed": employment_array.astype('bool'),
        "Attachment": attachment_array,
        "Location": pd.Categorical([country] * population, list(df.index)),
        "Migration": 0,
    }, columns=world_columns)
    return frame


def migrate_array(a, **kwargs):
    if len(a[a.Migration > MIGRATION_THRESHOLD]) == 0:
        return a.Location
#     np.random.seed(1000)
    migration_map = kwargs["migration_map"]
    countries = kwargs["countries"]
    for country, population in a.groupby("Location"):
        local_attraction = migration_map[country]
        local_attraction /= local_attraction.sum()
        migrants_num = len(population[population.Migration > MIGRATION_THRESHOLD])
        a.loc[(a.Country == country) & (a.Migration > MIGRATION_THRESHOLD),
              "Location"] = np.random.choice(countries,
                                             p=local_attraction,
                                             size=migrants_num,
                                             replace=True)
    return a.Location


def migrate_score(a, **kwargs):
    max_income = kwargs["max_income"]
    conflict_scores = kwargs["conflict"]
    max_conflict = kwargs["max_conflict"]
    conflict = conflict_scores.merge(a, left_index=True,
                                     right_on='Location')["Conflict"] / max_conflict
    return ((w1 * (1 + a.Income / -max_income) +
             (w2 * a.Attachment) +
             (w3 * conflict) +
             (w4 * a.Employed) + w5) / (w1+w2+w3+w4+w5)).astype('float32')


# def main():
def migration(w):
    
    global w1
    global w2
    global w3
    global w4
    global w5
    
    w1 = w[0]
    w2 = w[1]
    w3 = w[2]
    w4 = w[3]
    w5 = w[4]
    
    
#     np.random.seed(1000)
    globe = Globe(data.all(), processes=PROCESSES, splits=SPLITS)

    m = Manager()
    event = m.Event()

    globe.create_agents(generate_agents)

    globe.agents.Migration = globe.run_par(migrate_score, max_income=globe.agents.Income.max(),
                                           conflict=globe.df[["Conflict"]].sort_index(),
                                           max_conflict=globe.df.Conflict.max(),
                                           columns=["Income", "Employed", "Attachment", "Location"])

#     attractiveness = ((1 - globe.df["Conflict"] / globe.max_value("Conflict")) + (globe.df["GDP"] / globe.max_value("GDP")) + (1 - globe.df["Unemployment"] / globe.max_value("Unemployment")) + (1 - globe.df["Fertility"] / globe.max_value("Fertility")))

    attractiveness = (0.07170499) + (0.10970054) * (globe.df["Conflict"] / globe.max_value("Conflict") + (0.09184934) * globe.df["GDP"] / globe.max_value("GDP") + (-0.00126706) * globe.df["Unemployment"] / globe.max_value("Unemployment") + (-0.05770709) * globe.df["Fertility"] / globe.max_value("Fertility"))


#    attractiveness = (-0.04365079) + (-0.12915961) * (globe.df["Conflict"] / globe.max_value("Conflict") + (0.25397464) * globe.df["GDP"] / globe.max_value("GDP") + (-0.03426848) * globe.df["Unemployment"] / globe.max_value("Unemployment") + (0.06893584) * globe.df["Fertility"] / globe.max_value("Fertility"))
#    attractiveness[attractiveness < 0] = 0
#    print (attractiveness)
#    attractiveness.to_csv("attr.csv")

    #attractiveness = Wa * globe.df["Conflict"] +
    #                 Wb * globe.df["GDP"] +
    #                 Wc * globe.df["Unemployment"] +
    #                 Wd * globe.df["Fertility"]
    #                 -theta

    def neighbors(country):
        return globe.df[globe.df.index == country].iloc[0].neighbors

    migration_map = {}
    for country in globe.df.index:
        local_attraction = attractiveness.copy()
        local_attraction[local_attraction.index.isin(neighbors(country))] += 1
        migration_map[country] = local_attraction

    globe.agents["Location"] = globe.run_par(migrate_array, migration_map=migration_map,
                                             countries=globe.df.index,
                                             columns=["Country", "Location", "Migration"])

#     print("Migration model completed at a scale of {}:1.".format(int(1 / POPULATION_SCALE)))
    migrants = globe.agents[globe.agents.Country != globe.agents.Location]
#     print("There were a total of {} migrants.".format(len(migrants)))
#     print("There were a total of {} agents.".format(len(globe.agents)))
    changes = (globe.agents.Location.value_counts() -
               globe.agents.Country.value_counts()).sort_values()
#     print(changes.head())
#     print(changes.tail())
    #changes.to_csv("changes.csv")

#    netM = pd.read_csv("netM.csv")
#    for index, row in netM.iterrows():
#        if (row['NetMigration']).find("+AC0-") == -1:
#            row['NetMigration'] = int(float(row['NetMigration']) / 1000)
#        else:
#            row['NetMigration'] = int(float(row['NetMigration'].replace('+AC0-', '')) * (-1) / 1000)
#    #print(netM)

    global netM

    changesdf = changes.to_frame()
    changesdf.columns = ["Prdctn"]
    changesdf.index.name = "Code"
    changesdf.reset_index(inplace=True)
    #print(changesdf)
    combined = changesdf.merge(netM, how='left', on='Code')
    combined = combined.merge(pd.read_csv("df.csv", usecols=[0, 1]), how='left', on='Code')
    combined = combined.dropna()
#     print (combined)

    loss = 0
    for index, row in combined.iterrows():
        loss += ((row['NetMigration'] - row['Prdctn']) / row['Population']) ** (2)
        #print (loss)
#     print ()
    print ("Total loss:", loss)
    f = open('output.txt', 'a')
    print ("Total loss:", loss, file = f)
    f.close()
#     print ()

#     print("The potential migrants came from")
    migrants = globe.agents[globe.agents.Migration > MIGRATION_THRESHOLD]
#     print(migrants.Country.value_counts()[migrants.Country.value_counts().gt(0)])
#     return globe
    
    globe.pool.close()
    event.set()
    globe.pool.terminate()

    return loss

if __name__ == "__main__":
    DE(migration, 50, 10, 0, 25, 5)

