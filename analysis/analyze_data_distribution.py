import numpy as np
import pandas as pd
from utils.utils_data import load_data
from scipy import stats
from statsmodels.stats.proportion import proportions_chisquare
from os.path import join

# load data
data_table = load_data(select_feats=True, feats='last').reset_index()
raw_table = pd.read_csv('../../cohort_3/cohort3_demographics.csv', low_memory=False)
df_race_age = pd.read_csv('../../data_analysis/ethnicity.csv')
data_table = data_table.merge(df_race_age, how='left', on='admission_id')
data_table = data_table.set_index('admission_instance')

###############
# Using only complete cases
data_table = data_table[np.logical_and(data_table["race_ "] == 0, data_table["gender_U"] == 0)]
data_table = data_table[data_table.UN == False]
###############

# specimen types
df_specimen = pd.read_csv('../data_analysis/specimen_types_grouped.csv')
spec2type = dict(zip(df_specimen['0'], df_specimen['Grouping r=respiratory, u=urine, b=blood, o=other']))
score = {
    'r': 3,  # sputum/respiratory
    'u': 2,  # urine
    'o': 1,  # other
    'b': 0,  # blood
}
data_by_culture = pd.read_csv(join('../data_prepared/', 'data_by_culture.csv'))
data_by_culture['spec_type'] = data_by_culture['SPECIMEN'].apply(lambda x: score[spec2type[x]])
data_by_culture = data_by_culture[
    np.logical_and(
        data_by_culture.contain_GNB,
        data_by_culture.culture_level == data_by_culture.instance_level
    )
]
spec_type = data_by_culture.groupby('admission_instance')['spec_type'].max()


def select_subgroup(df_data, group):
    if group == 'All':
        return df_data.index.tolist()  # white
    elif group == 'Male':
        return df_data[df_data.Gender == 'Male'].index.tolist()
    elif group == 'White':
        return df_data[df_data.race_1 == 1].index.tolist()  # white
    elif group == 'Black':
        return df_data[df_data.race_2 == 1].index.tolist()  # black
    elif group == 'J15':
        return df_data[df_data.J15 > 0].index.tolist()
    elif group == 'A41':
        return df_data[df_data.A41 > 0].index.tolist()
    elif group == 'B96':
        return df_data[df_data.B96 > 0].index.tolist()
    elif group == 'Z16':
        return df_data[df_data.Z16 > 0].index.tolist()
    elif group == 'wo_B96/Z16/J15/A41':
        if_selected = ~(df_data.B96 + df_data.Z16 + df_data.J15 + df_data.A41).astype(bool)
        return df_data[if_selected].index.tolist()
    elif group == 'C81-96':
        # any comorbidities in C81-C96
        if_selected = (df_data.filter(regex='C8').sum(axis=1) + df_data.filter(regex='C9').sum(axis=1)).astype(bool)
        return df_data[if_selected].index.tolist()
    elif group == 'Z94':
        return df_data[df_data.Z94 > 0].index.tolist()
    elif group == 'K70.30':
        comorb = pd.read_csv('../../cohort_3/cohort3_diagnoses_for_comorbidities.csv', low_memory=False)
        pids = comorb[comorb.ICDX_DIAGNOSIS_CODE == 'K70.30'].reference_no.unique().tolist()
        instances = pd.read_csv('../data_analysis/instance_to_patient_id.csv')
        inst = instances[instances.patient_id.isin(pids)].admission_id.unique().tolist()
        return df_data[df_data.admission_id.isin(inst)].index.tolist()
    elif group == 'ESRD':
        comorb = pd.read_csv('../../cohort_3/cohort3_diagnoses_for_comorbidities.csv', low_memory=False)
        pids = comorb[comorb.ICDX_DIAGNOSIS_CODE == 'N18.6'].reference_no.unique().tolist()
        instances = pd.read_csv('../data_analysis/instance_to_patient_id.csv')
        inst = instances[instances.patient_id.isin(pids)].admission_id.unique().tolist()
        return df_data[df_data.admission_id.isin(inst)].index.tolist()
    elif group == 'COPD':
        comorb = pd.read_csv('../../cohort_3/cohort3_diagnoses_for_comorbidities.csv', low_memory=False)
        pids = comorb[comorb.ICDX_DIAGNOSIS_CODE == 'J44.9'].reference_no.unique().tolist()
        instances = pd.read_csv('../data_analysis/instance_to_patient_id.csv')
        inst = instances[instances.patient_id.isin(pids)].admission_id.unique().tolist()
        return df_data[df_data.admission_id.isin(inst)].index.tolist()
    elif group == 'Sepsis Shock':
        vasop = pd.read_csv('../data_prepared/data_last_vasop_by_instance.csv').drop(
            columns=['patient_id', 'admission_id'])
        vasop_names = vasop.columns[1:]
        if_selected = df_data[vasop_names].sum(axis=1).astype(bool)
        return df_data[if_selected].index.tolist()
    elif group == 'Intubation':
        return df_data[df_data.mechanical_ventilation > 0].index.tolist()
    elif group == 'Hospital_2574':
        return df_data[df_data.hospital_id_2574 > 0].index.tolist()
    elif group == 'Hospital_3148':
        return df_data[df_data.hospital_id_3148 > 0].index.tolist()
    elif group == 'Hospital_5107':
        return df_data[df_data.hospital_id_5107 > 0].index.tolist()
    elif group == 'Hospital_6729':
        return df_data[df_data.hospital_id_6729 > 0].index.tolist()
    elif group == 'Specimen_blood':
        spec_score = 0
        ids = spec_type[spec_type == spec_score].index.tolist()
        return list(set(df_data.index.tolist()) & set(ids))
    elif group == 'Specimen_urine':
        spec_score = 2
        ids = spec_type[spec_type == spec_score].index.tolist()
        return list(set(df_data.index.tolist()) & set(ids))
    elif group == 'Specimen_resp':
        spec_score = 3
        ids = spec_type[spec_type == spec_score].index.tolist()
        return list(set(df_data.index.tolist()) & set(ids))
    elif group == 'Specimen_other':
        spec_score = 1
        ids = spec_type[spec_type == spec_score].index.tolist()
        return list(set(df_data.index.tolist()) & set(ids))
    elif group == 'contain_GNB':
        df_data[df_data.index.isin(spec_type.index.tolist())].index.tolist()
    elif group == 'SS':
        return df_data[df_data.SS].index.tolist()
    elif group == 'RS':
        return df_data[df_data.RS].index.tolist()
    elif group == 'RR':
        return df_data[df_data.RR].index.tolist()


def gen_table_1(data_table, prefix=None):
    subgroups = ['N=', 'Age', 'Male', 'White', 'Black', 'J15', 'A41', 'B96', 'Z16', 'wo_B96/Z16/J15/A41', 'C81-96', 'Z94',
                 'K70.30', 'ESRD', 'COPD', 'Sepsis Shock', 'Intubation',
                 'Specimen_blood', 'Specimen_urine', 'Specimen_resp', 'Specimen_other']

    des = data_table.describe(include='all').transpose()

    # age
    print("Age: {:.0f} ({:.0f}, {:.0f})".format(des.loc['Age', '50%'],
                                                des.loc['Age', '25%'],
                                                des.loc['Age', '75%']))

    cohorts = {'all': data_table[data_table.UN == False],
               'SS': data_table[data_table.SS],
               'RS': data_table[data_table.RS],
               'RR': data_table[data_table.RR],
               'RR/RS': data_table[np.logical_or(data_table.RS, data_table.RR)]
               }

    rows = []
    for group in subgroups:
        row = {}
        row['Variable'] = group
        for cohort in cohorts.keys():
            if group == 'N=':
                row[cohort] = len(cohorts[cohort])
            elif group == 'Age':
                des = cohorts[cohort][['Age']].describe(include='all').transpose()
                row[cohort] = "{:.0f} ({:.0f}, {:.0f})".format(des.loc['Age', '50%'],
                                                               des.loc['Age', '25%'],
                                                               des.loc['Age', '75%'])
            else:
                if "Specimen" in group:
                    data_subcohort = cohorts[cohort]
                    data_subcohort = data_subcohort[data_subcohort.index.isin(spec_type.index.tolist())]
                    all = select_subgroup(data_subcohort, 'All')
                else:
                    all = select_subgroup(cohorts[cohort], 'All')
                inds = select_subgroup(cohorts[cohort], group)
                ratio = len(inds) / len(all) * 100
                row[cohort] = "{} ({:.1f}%)".format(len(inds), ratio)
        print(group, row)
        rows.append(row)

    table = pd.DataFrame.from_dict(rows, orient='columns')

    for group in subgroups:
        if group == 'N=':
            continue

        if group == 'Age':
            # one-way ANOVA
            ages_by_labels= [data_table.loc[data_table[label] == 1, 'Age'].values for label in ["SS", "RS", "RR"]]
            res = stats.f_oneway(*ages_by_labels)
            p_val = res.pvalue
        else:
            # chi-square test
            total_by_labels = [len(cohorts[label]) for label in ["SS", "RS", "RR"]]
            counts_by_labels = [int(table.loc[table.Variable == group, label].values[0].split(" ")[0]) for label in ["SS", "RS", "RR"]]
            chi2, p_val, cont_table = proportions_chisquare(count=counts_by_labels, nobs=total_by_labels)

        table.loc[table.Variable == group, 'p Value'] = p_val

    table.to_csv('../data_analysis/table_1_{}.csv'.format(prefix), index=False)


def gen_table_2(data_table, prefix=None):
    subgroups = ['N=', 'Age', 'Male', 'White', 'Black', 'J15', 'A41', 'B96', 'Z16', 'wo_B96/Z16/J15/A41', 'C81-96',
                 'Z94',
                 'K70.30', 'ESRD', 'COPD', 'Sepsis Shock', 'Intubation',
                 'Specimen_blood', 'Specimen_urine', 'Specimen_resp', 'Specimen_other',
                 'SS', 'RS', 'RR']

    data = data_table[data_table.UN == False]
    # hospitals = [col for col in data.columns if 'hospital' in col]
    hospitals = ['hospital_id_2574', 'hospital_id_3148', 'hospital_id_6729']
    cohorts = dict(zip(hospitals, [data[data[col].astype(bool)] for col in hospitals]))

    rows = []
    for group in subgroups:
        row = {}
        row['Variable'] = group
        for cohort in cohorts.keys():
            if group == 'N=':
                row[cohort] = len(cohorts[cohort])
            elif group == 'Age':
                des = cohorts[cohort][['Age']].describe(include='all').transpose()
                row[cohort] = "{:.0f} ({:.0f}, {:.0f})".format(des.loc['Age', '50%'],
                                                               des.loc['Age', '25%'],
                                                               des.loc['Age', '75%'])
            else:
                if "Specimen" in group:
                    data_subcohort = cohorts[cohort]
                    data_subcohort = data_subcohort[data_subcohort.index.isin(spec_type.index.tolist())]
                    all = select_subgroup(data_subcohort, 'All')
                else:
                    all = select_subgroup(cohorts[cohort], 'All')
                inds = select_subgroup(cohorts[cohort], group)
                ratio = len(inds) / len(all) * 100
                row[cohort] = "{} ({:.1f}%)".format(len(inds), ratio)
        print(group, row)
        rows.append(row)

    table = pd.DataFrame.from_dict(rows, orient='columns')

    for group in subgroups:
        if group == 'N=':
            continue
        if group == 'Age':
            # one-way ANOVA
            ages_by_hospitals = [data_table.loc[data_table[hosp] == 1, 'Age'].values for hosp in hospitals]
            res = stats.f_oneway(*ages_by_hospitals)
            p_val = res.pvalue
        else:
            # chi-square test
            total_by_hospitals = [len(cohorts[hosp]) for hosp in hospitals]
            counts_by_hospitals = [int(table.loc[table.Variable == group, hosp].values[0].split(" ")[0]) for hosp in hospitals]
            chi2, p_val, cont_table = proportions_chisquare(count=counts_by_hospitals, nobs=total_by_hospitals)

        table.loc[table.Variable == group, 'p Value'] = p_val

    table.to_csv('../data_analysis/table_2_{}.csv'.format(prefix), index=False)


gen_table_1(data_table[data_table.infection_id == 0], prefix='initial')
gen_table_1(data_table[data_table.infection_id > 0], prefix='subsequent')

gen_table_2(data_table[data_table.infection_id == 0], prefix='initial')
gen_table_2(data_table[data_table.infection_id > 0], prefix='subsequent')




