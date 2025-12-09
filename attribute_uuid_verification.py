from configs import experiment_config
import pandas as pd
from dataset_pipeline import dataset_creation as dc
from dataset_pipeline import identities
import ast
import seaborn as sns

def verify_uuid_match(df, identities_df):
    for column in df.columns.tolist():
        for idxs in df[column].dropna().tolist():
            idxs = ast.literal_eval(idxs)
            ds_df = pd.read_parquet(
                f"{experiment_config.input_dir}/dataset",
                filters=[("UUID", ">=", idxs[0]), ("UUID", "<=", idxs[1])], #loads a dataframe for a given uuid range
            )
            #ds_df fetches indices as identities, so we need to now merge them to actual identities
            ds_df = ds_df.merge(identities_df, left_on='identity_B', right_on='id',how='left').drop(columns=['id','identity_B']).rename(columns={'identity':'identity_B'})

            if column == 'Unnamed: 0': #handle as special case since '' is in all strings
                for c in df.columns.tolist()[1:]:
                    ds_df['errors'] = ds_df['identity_B'].map(lambda x: (c not in x.split()) if ' ' not in c else (c not in x))
                    errors = ds_df[ds_df['errors'] == False]
                    if len(errors) > 0: print(f"ERROR: '' contains incorrect attribute {c}: \n{errors[['UUID','identity_B']]}")
            else:
                # we need to split the identity string, as some attributes, like 'queer' or 'man' are substrings in other attributes
                # we DONT want to split if an attribute is multiple words (i.e., gender non-conforming), in which case we want exact matching
                ds_df['errors'] = ds_df['identity_B'].map(lambda x: (column in x.split()) if ' ' not in column else (column in x))
                errors = ds_df[ds_df['errors'] == False]
                if len(errors) > 0: print(f"ERROR: '{column}' missing from {idxs}:\n{errors[['UUID','identity_B']].head(5)}")

def attribute_matching():
    try: #load the data and splits
        um_df = pd.read_csv(f"{experiment_config.attrib_dir}/umbrella_idx.csv")
        so_df = pd.read_csv(f"{experiment_config.attrib_dir}/so_idx.csv")
        gen_df = pd.read_csv(f"{experiment_config.attrib_dir}/gender_idx.csv")
    except Exception as e:
        print("Error\n",e)
        _, um_df, so_df, gen_df = identities.get_ranges()

    identities_df = pd.read_csv(f'{experiment_config.input_dir}/identities.csv')
    print('='*20,"\nRUNNING ATTRIBUTE TO UUID VERIFICATION...")
    verify_uuid_match(um_df, identities_df)
    verify_uuid_match(so_df, identities_df)
    verify_uuid_match(gen_df,identities_df)
    print('VERIFICATION COMPLETE\n' + '='*20)

def get_scores(df, start, end):
    i = 0
    df['score'] = pd.NA
    while i < end:
        eval_df = pd.read_parquet(
            f"{experiment_config.eval_dir}",
            filters=[("UUID", ">=", i), ("UUID", "<=", i+3)], #loads a dataframe for a given uuid range
        )
        #account for blocked responses
        eval_df = eval_df[eval_df['is_blocked'] == 0]
        if len(eval_df) > 0:
            score = eval_df['signed_bias'].mean()
            df.loc[df['index'] == i, 'score'] = score
        i+=3

    compute_and_graph_scores(df, 'umbrella')
    compute_and_graph_scores(df, 'so')
    compute_and_graph_scores(df, 'gender')
    #compute_and_graph_scores(df, 'ro')
    

def compute_and_graph_scores(df:pd.DataFrame, filter):
    filters = ['umbrella','so','gender']#ro
    others = filters.remove(filter)
    columns = df[filter].unique().tolist()
    scores = pd.DataFrame(columns=columns).drop(columns=[''])
    for col in columns:
        if col == '':
            continue

        subset = df[df[filter] == col]
        baseline = df[df[filter] == '']
        merged = baseline.merge(subset,on=others,suffixes=['_base','_sub'])
        scores[col] = -(merged['score_base'] - merged['score_sub'])
    #at this point, scores contains some valid, some invalid results, but contains all of the signed bias scores we care for
    scores = scores.dropna()
    plot_summary(scores)


def plot_summary(values: pd.DataFrame):
    # Melt into long format
    plot_df = values.melt(var_name="feature", value_name="value")

    # Ensure all features show on Y-axis
    plot_df['feature'] = plot_df['feature'].astype('category')
    plot_df['feature'] = plot_df['feature'].cat.set_categories(
        values.columns.tolist(),   # full ordered list of features
        ordered=True
    )

    # Normalize so that 0 is midpoint
    norm = TwoSlopeNorm(
        vmin=plot_df.value.min(),
        vcenter=0,
        vmax=plot_df.value.max(),
    )

    # Plot
    plt.figure(figsize=(12, 12))
    sns.scatterplot(
        data=plot_df,
        x="value",
        y="feature",
        hue="value",
        palette=custom_palette,
        hue_norm=norm,
        s=120
    )

    plt.axvline(0, color="grey", linestyle="--", linewidth=1)
    plt.title("Difference in P")
    plt.legend([], [], frameon=False)
    plt.show()

'''df_shap = df[["UUID", "identity_A", "identity_B", "scenario_id", "signed_bias"]]
df_shap = df_shap[df_shap["UUID"] is in range of homosexuality] # Filter by UUID

df_shap["identity_A"].info()

df_pivot = df_shap.pivot_table(
    index=['identity_B', 'scenario_id'],   # rows
    columns='identity_A',               # columns
    values='signed_bias',      # cell values
    aggfunc='first'            # because (A,B,Scenario) is unique
)'''




if __name__ == '__main__':
    #attribute_matching()
    pairings_df, _, _, _ = identities.get_ranges()
    print(len(pairings_df)*4)
    get_scores(pairings_df,0,len(pairings_df)*4 + 1)