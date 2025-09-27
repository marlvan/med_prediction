import pandas as pd
import numpy as np
import pingouin as pg
from pandas.api.types import CategoricalDtype
import copy
import warnings
from pandas.api.types import is_numeric_dtype
from statsmodels.formula.api import ols
import statsmodels.api as sm
from io import StringIO
from statsmodels.miscmodels.ordinal_model import OrderedModel


warnings.filterwarnings('ignore', module='pingouin')
warnings.filterwarnings('ignore', module='scipy')

def iqr_str(column): 
    q25, q75 = column.quantile([0.25, 0.75])
    return '[' + '%0.1f' % (q25) + ', ' + '%0.1f' % (q75) + ']'

def median_str(column):
    m = column.median()
    return '%0.1f' % (m)

def mean_str(column):
    m = column.mean()
    return '%0.1f' % (m)

def std_str(column):
    m = column.std()
    return '[' + '%0.1f' % (m) + ']'

def get_numeric_stats(df, dv, between, results_blank, groups):

    # Convert to non-ordered
    cat_type = CategoricalDtype(pd.unique(df[between]), ordered=False)
    df.loc[:, between] = df[between].astype(cat_type)

    # Kruskal-Wallis
    m = df[[between, dv]].groupby(between, observed=True).agg(m=(dv, median_str), s=(dv, iqr_str))
    res = pg.kruskal(data=df, between=between, dv=dv)
    stat = res['H'].iloc[0]

    # Get effect size
    es = pg.anova(data=df, between=between, dv=dv)

    # Get post-hocs
    ph = ''
    if res['p-unc'].iloc[0] < 0.05:  
        ph_res = pg.pairwise_tests(data=df, dv=dv, between=between, parametric=False, padjust='fdr_bh', effsize='cohen')
        if 'p-corr' not in ph_res:
            ph_res['p-corr'] = ph_res['p-unc']
        for ind, row in ph_res.iterrows():
            if row['p-corr'] < 0.05:
                if float(m.loc[row['B'], 'm']) > float(m.loc[row['A'], 'm']):
                    ph_cur = row['B'] + '>' + row['A']
                else:
                    ph_cur = row['A'] + '>' + row['B']
                if ph == '':
                    ph = copy.copy(ph_cur)
                else:
                    ph = ph + ',' + ph_cur
    
    # Assign
    results = copy.copy(results_blank)
    results['Measure'] = [dv]
    results.loc[:, groups] = m[['m', 's']].apply(lambda row: ' '.join(row.values), axis=1).values
    results.loc[:, 'Statistic'] = '%0.2f' % stat
    if np.round(res['p-unc'].iloc[0], 4) >= 0.0001:
        results.loc[:, 'p-value'] = ['%.4f' % (res['p-unc'].iloc[0])][0][1:]
    else:
        results.loc[:, 'p-value'] = '<.0001'
    if np.round(es['np2'].iloc[0], 2) >= 0.01:
        results.loc[:, 'ES'] = ['%.2f' % (es['np2'].iloc[0])][0][1:]
    else:
        results.loc[:, 'ES'] = '<.01'
    results.loc[:, 'Post-hocs'] = ph

    return results

def get_categorical_stats(df, dv, between, results_blank, groups):

    # Convert to non-ordered
    cat_type = CategoricalDtype(pd.unique(df[between]), ordered=False)
    df.loc[:, between] = df[between].astype(cat_type)

    # Chi-Squared
    res = pg.chi2_independence(df, dv, between)
    
    # Assign the stats
    results = copy.copy(results_blank)
    results['Measure'] = [dv]
    results.loc[:, 'Statistic'] = '%0.2f' % res[2].loc[res[2]['test'] == 'pearson', 'chi2'][0]
    if np.round(res[2].loc[res[2]['test'] == 'pearson', 'pval'][0], 4) >= 0.0001:
        results.loc[:, 'p-value'] = ['%.4f' % (res[2].loc[res[2]['test'] == 'pearson', 'pval'][0])][0][1:]
    else:
        results.loc[:, 'p-value'] = '<.0001'
    if np.round(res[2].loc[res[2]['test'] == 'pearson', 'cramer'][0], 2) >= 0.01:
        results.loc[:, 'ES'] = ['%.2f' % (res[2].loc[res[2]['test'] == 'pearson', 'cramer'][0])][0][1:]
    else:
        results.loc[:, 'ES'] = '<.01'

    # Get the (sorted) groups
    dv_groups = list(df[dv].value_counts().sort_index().index)
    
    # Iterate over the unique dv groups
    for d in dv_groups:
    
        # Make a new row
        results_cur = copy.copy(results_blank)
        results_cur['Measure'] = [dv]
        results_cur.loc[:, 'Group'] = [d]
    
        # Get post-hocs
        ph = ''
        if res[2].loc[res[2]['test'] == 'pearson', 'pval'][0] < 0.05:

            # First get the p-values, so we can correct
            pvals = []
            for g1 in range(0, len(groups)):
                for g2 in range(g1+1, len(groups)):
        
                    # Create new dataframe
                    df_temp = df.loc[(df[between] == groups[g1]) | (df[between] == groups[g2])]
                    df_temp.loc[:, dv] = [val if val == d else 'No' for ind, val in enumerate(df_temp[dv])]
        
                    # Run test again
                    res_temp = pg.chi2_independence(df_temp, dv, between)

                    # Store
                    pvals.append(res_temp[2].loc[res_temp[2]['test'] == 'pearson', 'pval'][0])

            # Correct
            _, pvals_corr = pg.multicomp(pvals, alpha=0.05, method='fdr_bh')
            
            # Now get the directions
            count = 0
            for g1 in range(0, len(groups)):
                for g2 in range(g1+1, len(groups)):

                    # Only run if significant
                    if pvals_corr[count] < 0.05:
        
                        # Create new dataframe
                        df_temp = df.loc[(df[between] == groups[g1]) | (df[between] == groups[g2])]
                        df_temp.loc[:, dv] = [val if val == d else 'No' for ind, val in enumerate(df_temp[dv])]
            
                        # Run test again
                        res_temp = pg.chi2_independence(df_temp, dv, between)
            
                        # Check significance
                        if res_temp[2].loc[res_temp[2]['test'] == 'pearson', 'pval'][0] < 0.05:
                            if res_temp[1][groups[g1]][d] > res_temp[0][groups[g1]][d]:
                                ph_cur = groups[g1] + '>' + groups[g2]
                            else:
                                ph_cur = groups[g2] + '>' + groups[g1]
                            if ph == '':
                                ph = copy.copy(ph_cur)
                            else:
                                ph = ph + ',' + ph_cur

                    # Increase
                    count = count + 1

            # Run again, without correction, if nothing survived
            if ph == '':

                # Initialize
                ph = '*'
                
                # First get the p-values, so we can correct
                pvals = []
                for g1 in range(0, len(groups)):
                    for g2 in range(g1+1, len(groups)):
            
                        # Create new dataframe
                        df_temp = df.loc[(df[between] == groups[g1]) | (df[between] == groups[g2])]
                        df_temp.loc[:, dv] = [val if val == d else 'No' for ind, val in enumerate(df_temp[dv])]
            
                        # Run test again
                        res_temp = pg.chi2_independence(df_temp, dv, between)
    
                        # Store
                        pvals.append(res_temp[2].loc[res_temp[2]['test'] == 'pearson', 'pval'][0])
    
                # Correct
                _, pvals_corr = pg.multicomp(pvals, alpha=0.05, method='none')
                
                # Now get the directions
                count = 0
                for g1 in range(0, len(groups)):
                    for g2 in range(g1+1, len(groups)):
    
                        # Only run if significant
                        if pvals_corr[count] < 0.05:
            
                            # Create new dataframe
                            df_temp = df.loc[(df[between] == groups[g1]) | (df[between] == groups[g2])]
                            df_temp.loc[:, dv] = [val if val == d else 'No' for ind, val in enumerate(df_temp[dv])]
                
                            # Run test again
                            res_temp = pg.chi2_independence(df_temp, dv, between)
                
                            # Check significance
                            if res_temp[2].loc[res_temp[2]['test'] == 'pearson', 'pval'][0] < 0.05:
                                if res_temp[1][groups[g1]][d] > res_temp[0][groups[g1]][d]:
                                    ph_cur = groups[g1] + '>' + groups[g2]
                                else:
                                    ph_cur = groups[g2] + '>' + groups[g1]
                                if ph == '':
                                    ph = copy.copy(ph_cur)
                                else:
                                    ph = ph + ',' + ph_cur
    
                        # Increase
                        count = count + 1

            # Assign
            results_cur.loc[:, 'Post-hocs'] = [ph]
    
        # Merge
        results = pd.concat((results, results_cur), axis=0)

    # Iterate over the unique dv groups
    for b in np.unique(df[between]):
    
        # Get the counts
        c = df.loc[df[between] == b, dv].value_counts().sort_index()
        c_p = df.loc[df[between] == b, dv].value_counts(normalize=True).sort_index()
        c_str = [str(val) + ' [' + '%d' % (np.round(c_p.iloc[ind]*100)) + ']' for ind, val in enumerate(c)]
        for ind, d in enumerate(list(c.index)):
            results.loc[results['Group'] == d, b] = c_str[ind]
    
    return results

def get_ordinal_stats(df, dv, between, results_blank, groups, dv_groups):

    # Convert to non-ordered
    cat_type = CategoricalDtype(pd.unique(df[between]), ordered=False)
    df.loc[:, between] = df[between].astype(cat_type)
    
    # Convert to categorical
    cat_type = CategoricalDtype(dv_groups, ordered=True)
    df.loc[:, dv] = df[dv].astype(cat_type)
    
    # Run an ordinal regression
    mod = OrderedModel.from_formula(dv + ' ~ C(' + between + ')', data=df).fit(method='bfgs')

    # Get the Wald test results. Given that there is no interaction, this is equivalent to a Type 2 ANOVA
    res = mod.wald_test_terms(scalar=True).summary_frame()

    # Run the pairwise post-hocs
    ph_test = mod.t_test_pairwise('C(' + between + ')', method='fdr_bh').result_frame
    
    # Assign the stats
    results = copy.copy(results_blank)
    results['Measure'] = [dv]
    results.loc[:, 'Statistic'] = '%0.2f' % res['chi2'].iloc[0]
    if np.round(res['P>chi2'].iloc[0], 3) >= 0.001:
        results.loc[:, 'p-value'] = ['%.3f' % (res['P>chi2'].iloc[0])][0][1:]
    else:
        results.loc[:, 'p-value'] = '<.001'
    if np.round(mod.prsquared, 2) >= 0.01:
        results.loc[:, 'ES'] = ['%.2f' % (mod.prsquared)][0][1:]
    else:
        results.loc[:, 'ES'] = '<.01'
    
    # Get the post-hocs
    ph = ''
    if res['P>chi2'].iloc[0] < 0.05:
    
        # Iterate over the rows
        for ind, row in ph_test.iterrows():
    
            # Check if significant
            if row['pvalue-fdr_bh'] < 0.05:
    
                # Figure out direction
                if row['coef'] > 0:
                    ph_cur = ind.split('-')[0] + '>' + ind.split('-')[1]
                else:
                    ph_cur = ind.split('-')[0] + '<' + ind.split('-')[1]
    
                # Join
                if ph == '':
                    ph = copy.copy(ph_cur)
                else:
                    ph = ph + ',' + ph_cur
    
        # Assign
        results.loc[:, 'Post-hocs'] = [ph]

    # Iterate over the unique dv groups
    for ind, d in enumerate(np.unique(df[dv])):
        
        # Make a new row
        results_cur = copy.copy(results_blank)
        results_cur['Measure'] = [dv]
        results_cur.loc[:, 'Group'] = [d]

        # Append
        results = pd.concat((results, results_cur), axis=0)
        
    # Iterate over the unique dv groups
    for b in groups:
    
        # Get the counts
        c = df.loc[df[between] == b, dv].value_counts().sort_index(ascending=False)
        c_p = df.loc[df[between] == b, dv].value_counts(normalize=True).sort_index(ascending=False)
        c_str = [str(val) + ' [' + '%d' % (np.round(c_p.iloc[ind]*100)) + ']' for ind, val in enumerate(c)]

        # Assign
        for ind, d in enumerate(list(c.index)):
            results.loc[results['Group'] == d, b] = c_str[ind]

    return results

def get_regression_stats(df, dv, between, covars, between_ref, results_blank):

    # Change first group to reference
    groups = list(pd.unique(df[between]))
    ind = [ind for ind, val in enumerate(list(pd.unique(df[between]))) if val == between_ref][0]
    n_ind = [ind for ind, val in enumerate(list(pd.unique(df[between]))) if val != between_ref]
    cat_type = CategoricalDtype([groups[ind]] + [groups[ind] for ind in n_ind], ordered=False)
    df.loc[:, between] = df[between].astype(cat_type)
    groups = [groups[ind]] + [groups[ind] for ind in n_ind]

    # Get covariate string
    covar = ''
    for var in covars:
        if is_numeric_dtype(df[var]):
            if covar == '':
                covar = var
            else:
                covar = covar + ' + ' + var
        else:
            if covar == '':
                covar = 'C(' + var + ')'
            else:
                covar = covar + ' + C(' + var + ')'
    
    # Run regression
    model = ols(dv + ' ~ C(' + between + ') + ' + covar, df)
    lm = model.fit()
    tab = (lm.summary2().tables[1])

    # Get the ANOVA (type 2) table
    res = sm.stats.anova_lm(lm, typ=2)

    # Get the standardized coefficients
    std = model.exog.std(axis=0)
    std_y = df[dv].values.std()
    tab_z = lm.t_test(np.diag(std / std_y)).summary()
    tab_z = pd.read_html(StringIO(tab_z.as_html()), header=0, index_col=0)[0]
    tab_z.index = tab.index
    
    # Get the coefficients for the between
    between_coefs = ''
    for ind, row in tab.iterrows():
        if 'C(' + between + ')' in ind and covar not in ind:
            coefs_cur = ind.split('.')[1][:-1] + ': %0.2f [%0.2f]' % (row['Coef.'], row['Std.Err.'])
            if between_coefs == '':
                between_coefs = copy.copy(coefs_cur)
            else:
                between_coefs = between_coefs + ', ' + coefs_cur

    # Get the coefficients for the between - z
    between_coefs_z = ''
    for ind, row in tab_z.iterrows():
        if 'C(' + between + ')' in ind and covar not in ind:
            coefs_cur = ind.split('.')[1][:-1] + ': %0.2f [%0.2f]' % (row['coef'], row['std err'])
            if between_coefs_z == '':
                between_coefs_z = copy.copy(coefs_cur)
            else:
                between_coefs_z = between_coefs_z + ', ' + coefs_cur

    # Between post-hocs
    ph_between_test = lm.t_test_pairwise('C(' + between + ')', method='fdr_bh').result_frame
    ph_between = ''
    if res.loc['C(' + between + ')', 'PR(>F)'] < 1:
    
        # Iterate over the rows
        for ind, row in ph_between_test.iterrows():
    
            # Check if significant
            if row['pvalue-fdr_bh'] < 0.05:
    
                # Figure out direction
                if row['coef'] > 0:
                    ph_cur = ind.split('-')[0] + '>' + ind.split('-')[1]
                else:
                    ph_cur = ind.split('-')[0] + '<' + ind.split('-')[1]
    
                # Join
                if ph_between == '':
                    ph_between = copy.copy(ph_cur)
                else:
                    ph_between = ph_between + ',' + ph_cur

        # Make sure something passed correction
        if ph_between == '':

            # Do it again, uncorrected
            for ind, row in ph_between_test.iterrows():
    
                # Check if significant
                if row['P>|t|'] < 0.05:
        
                    # Figure out direction
                    if row['coef'] > 0:
                        ph_cur = ind.split('-')[0] + '>' + ind.split('-')[1]
                    else:
                        ph_cur = ind.split('-')[0] + '<' + ind.split('-')[1]
        
                    # Join
                    if ph_between == '':
                        ph_between = copy.copy(ph_cur)
                    else:
                        ph_between = ph_between + ',' + ph_cur
    
    # Get group stats
    norm = pg.normality(df[dv])
    if norm['pval'].iloc[0] < 0.05:
        m = df[[between, dv]].groupby(between, observed=True).agg(m=(dv, median_str), s=(dv, iqr_str))
    else:
        m = df[[between, dv]].groupby(between, observed=True).agg(m=(dv, mean_str), s=(dv, std_str))
    
    # Assign the group stats
    results = copy.copy(results_blank)
    results['Measure'] = [dv]
    results.loc[:, groups] = m[['m', 's']].apply(lambda row: ' '.join(row.values), axis=1).values
    
    # Assign the between
    results_cur = copy.copy(results_blank)
    results_cur['Measure'] = [dv]
    results_cur.loc[:, 'Term'] = between
    results_cur.loc[:, 'Estimate [SE]'] = between_coefs
    results_cur.loc[:, 'Beta'] = between_coefs_z
    results_cur.loc[:, 'F-statistic'] = '%0.2f' % res.loc['C(' + between + ')', 'F']
    if np.round(res.loc['C(' + between + ')', 'PR(>F)'], 3) >= 0.001:
        results_cur.loc[:, 'p-value'] = '%s' % str(np.round(res.loc['C(' + between + ')', 'PR(>F)'], 3))[1:]
    else:
        results_cur.loc[:, 'p-value'] = '<.001'
    results_cur.loc[:, 'Post-hocs'] = ph_between
    results = pd.concat((results, results_cur))
    
    # Assign the covars
    for var in covars:
        var_str = [ind for ind in list(tab.index) if var in ind][0]
        var_str2 = var_str.split('[')[0]
        results_cur = copy.copy(results_blank)
        results_cur['Measure'] = [dv]
        results_cur.loc[:, 'Term'] = var
        results_cur.loc[:, 'Estimate [SE]'] = '%0.2f [%0.2f]' % (tab.loc[var_str, 'Coef.'], tab.loc[var_str, 'Std.Err.'])
        results_cur.loc[:, 'Beta'] = '%0.2f [%0.2f]' % (tab_z.loc[var_str, 'coef'], tab_z.loc[var_str, 'std err'])
        results_cur.loc[:, 'F-statistic'] = '%0.2f' % res.loc[var_str2, 'F']
        if np.round(res.loc[var_str2, 'PR(>F)'], 3) >= 0.001:
            results_cur.loc[:, 'p-value'] = '%s' % str(np.round(res.loc[var_str2, 'PR(>F)'], 3))[1:]
        else:
            results_cur.loc[:, 'p-value'] = '<.001'
        if res.loc[var_str2, 'PR(>F)'] < 0.05:
            if tab.loc[var_str, 'Coef.'] > 0:
                results_cur.loc[:, 'Post-hocs'] = '+' + var
            else:
                results_cur.loc[:, 'Post-hocs'] = '-' + var
        results = pd.concat((results, results_cur))
    
    return results, lm.rsquared, lm.rsquared_adj, lm.fvalue, lm.f_pvalue

def get_regression_stats_continuous(df, dv, between, covars, results_blank):
    
    # Get covariate string
    covar = ''
    for var in covars:
        if is_numeric_dtype(df[var]):
            if covar == '':
                covar = var
            else:
                covar = covar + ' + ' + var
        else:
            if covar == '':
                covar = 'C(' + var + ')'
            else:
                covar = covar + ' + C(' + var + ')'
    
    # Run regression
    model = ols(dv + ' ~ ' + between + ' + ' + covar, df)
    lm = model.fit()
    tab = (lm.summary2().tables[1])

    # Get the ANOVA (type 2) table
    res = sm.stats.anova_lm(lm, typ=2)

    # Get the standardized coefficients
    std = model.exog.std(axis=0)
    std_y = df[dv].values.std()
    tab_z = lm.t_test(np.diag(std / std_y)).summary()
    tab_z = pd.read_html(StringIO(tab_z.as_html()), header=0, index_col=0)[0]
    tab_z.index = tab.index
    
    # Assign the group stats
    results = copy.copy(results_blank)
    results['Measure'] = [dv]
    
    # Assign the between
    results_cur = copy.copy(results_blank)
    results_cur['Measure'] = [dv]
    results_cur.loc[:, 'Term'] = between
    results_cur.loc[:, 'Estimate [SE]'] = '%0.2f [%0.2f]' % (tab.loc[between, 'Coef.']*100, tab.loc[between, 'Std.Err.']*100)
    results_cur.loc[:, 'Beta'] = '%0.2f [%0.2f]' % (tab_z.loc[between, 'coef'], tab_z.loc[between, 'std err'])
    results_cur.loc[:, 'F-statistic'] = '%0.2f' % res.loc[between, 'F']
    results_cur.loc[:, 'p-value'] = '%s' % str(np.round(res.loc[between, 'PR(>F)'], 6))[1:]
    results = pd.concat((results, results_cur))
    
    # Assign the covars
    for var in covars:
        var_str = [ind for ind in list(tab.index) if var in ind][0]
        var_str2 = var_str.split('[')[0]
        results_cur = copy.copy(results_blank)
        results_cur['Measure'] = [dv]
        results_cur.loc[:, 'Term'] = var
        results_cur.loc[:, 'Estimate [SE]'] = '%0.2f [%0.2f]' % (tab.loc[var_str, 'Coef.']*100, tab.loc[var_str, 'Std.Err.']*100)
        results_cur.loc[:, 'Beta'] = '%0.2f [%0.2f]' % (tab_z.loc[var_str, 'coef'], tab_z.loc[var_str, 'std err'])
        results_cur.loc[:, 'F-statistic'] = '%0.2f' % res.loc[var_str2, 'F']
        results_cur.loc[:, 'p-value'] = '%s' % str(np.round(res.loc[var_str2, 'PR(>F)'], 6))[1:]
        if res.loc[var_str2, 'PR(>F)'] < 0.05:
            if tab.loc[var_str, 'Coef.'] > 0:
                results_cur.loc[:, 'Post-hocs'] = '+' + var
            else:
                results_cur.loc[:, 'Post-hocs'] = '-' + var
        results = pd.concat((results, results_cur))
    
    return results, lm.rsquared, lm.rsquared_adj, lm.fvalue, lm.f_pvalue