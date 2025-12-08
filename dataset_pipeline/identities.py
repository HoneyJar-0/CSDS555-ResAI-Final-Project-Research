#from configs import experiment_config

def get_queer_attributes():
    '''
    Takes a string of identities from the GenderCensus and organizes the identities into one of: 
    Umbrella, Gender Identity, Sexual Orientation, or Romantic Orientation.

    *Disclaimer: We are aware that some of these identities could belong in other categories. For the sake of simplicity
    and proof-of-concept, we will use these concrete categories. Further experimentation can involve overlapping.
    '''
    unique = ['nonqueer' ]
    '''
    removed identities due to combinatoral explosion:
     LGBT person, LGBTQ person, LGBTQI person, LGBTQIA person, catgender
    '''
    # Changed due to time constraints
    unsorted_OLD = 'queer, LGBTQIA+, cisgender, man, woman, male, female, gender conforming, nonbinary, enby, gender non-conforming, polygender, agender, genderless, genderfluid, xenogender, transgender, transsexual, trans, transwoman, transman, genderqueer, pangender, demigender, intersexual, intersex, androgynous, gay, lesbian, bisexual, pansexual, straight, heterosexual, homosexual, asexual, demisexual, homoromantic, biromantic, panromantic, aromantic, heteroromantic'
    
    unsorted = 'queer, LGBTQIA+, cisgender, man, woman, nonbinary, gender non-conforming, polygender, agender, genderfluid, transgender, transwoman, transman, genderqueer, androgynous, gay, lesbian, bisexual, pansexual, straight, heterosexual, homosexual, asexual, homoromantic, biromantic, panromantic, aromantic, heteroromantic'
    unsorted = unsorted.split(',')
    umbrella = [''] #add empty string to each category to include the "not affiliated" identity; i.e., "I don't identify/describe myself with this"
    gender = ['']
    so = ['']
    ro = ['']
    flag = 'umbrella'

    for attribute in unsorted:
        attribute = attribute.split('person')[0].strip() #remove "person" since this will be in the base prompt
        if flag == 'umbrella':
            umbrella.append(attribute)
        elif flag == 'gender':
            gender.append(attribute)
        elif flag == 'so':
            so.append(attribute)
        else:
            ro.append(attribute)

        if attribute == 'LGBTQIA+':
            flag = 'gender'
        elif attribute == 'androgynous':
            flag = 'so'
        elif attribute == 'asexual1111':
            flag = 'ro'
    return umbrella, gender, so, ro

def attribute_pairing(umbrella, gender, so, ro):
    permutations = []
    for u in umbrella: #RIP Never-Nesters :3
        for s in so:
            for r in ro:
                for g in gender:
                    do_not_append_person = ['man','male','woman','female'] #grammar; "I am a man person" is not grammatically correct
                    if not any(substring in g.lower() for substring in do_not_append_person):
                        g = (g + ' person') if g != '' else 'person'

                    identity = ""
                    for attr in [u,s,r,g]:
                        if attr != '':
                            identity = identity + attr + ' '
                    # Now we filter out nonqueer identities from the permutations.
                    if not (s == 'heterosexual' and r == 'heteroromantic' and (g == 'man' or g == 'male' or g == 'woman' or g == 'female')):
                        permutations.append(identity.strip())
    for gender in ['man','woman','male','female','person']:
        permutations.append('nonqueer ' + gender)
    permutations[-1] = permutations[-1].strip()
    return permutations

def save_identities_to_file(identities):
    with open(f"{experiment_config.input_dir}/identities.csv", 'w') as fp:
        fp.write("id,identity\n")
        for i, ident in enumerate(identities):
            fp.write(f"{i},{ident}\n")

def identity_pipeline():
    umbrella, gender, so, ro = get_queer_attributes()
    print(f"Number entries:\nUmbrella: {len(umbrella)}, Gender: {len(gender)}, SO: {len(so)}, RO: {len(ro)}")
    print(umbrella, gender, so, ro)
    #permutations = attribute_pairing(umbrella, gender, so, ro)
    #save_identities_to_file(permutations)

def get_ranges():
    import pandas as pd
    umbrella, gender, so, ro = get_queer_attributes()
    pairings_df = pd.DataFrame(columns=['umbrella','so','ro','gender'])

    for u in umbrella: #RIP Never-Nesters 2 :3
        for s in so:
            for r in ro:
                for g in gender:
                    if not (s == 'heterosexual' and r == 'heteroromantic' and (g == 'man' or g == 'male' or g == 'woman' or g == 'female')):
                        temp = {
                            'umbrella':[u],
                            'so':[s],
                            'ro':[r],
                            'gender':[g]
                        }
                        pairings_df = pd.concat([pairings_df, pd.DataFrame(temp)], ignore_index = True)
    for g in ['man','woman','male','female','']: #clean up crew: female and male were cut
        temp = {
            'umbrella':['nonqueer'],
            'so':[''],
            'ro':[''],
            'gender':[g]
        }
        pairings_df = pd.concat([pairings_df,pd.DataFrame(temp)],ignore_index=True)
    um_df = pd.DataFrame()
    so_df = pd.DataFrame()
    gen_df = pd.DataFrame()

    pairings_df = pairings_df.reset_index()
    pairings_df['index'] = pairings_df['index'].apply(lambda x: x*4)
    pairings_df['end'] = pairings_df['index'].apply(lambda x: x + 3)
    umbrella.append('nonqueer')
    for u in umbrella:
        subset = pairings_df[pairings_df['umbrella'] == u]
        um_df[u] = pd.Series(idx_grouping(subset['index'].tolist(), subset['end'].tolist()))
        print(f"umbrella {u} handled")
    for s in so:
        subset = pairings_df[pairings_df['so'] == s]
        so_df[s] = pd.Series(idx_grouping(subset['index'].tolist(), subset['end'].tolist()))
        print(f"so {s} handled")
    for g in gender:
        subset = pairings_df[pairings_df['gender'] == g]
        gen_df[g] = pd.Series(idx_grouping(subset['index'].tolist(), subset['end'].tolist()))
        print(f"gender {g} handled")
    
    um_df.to_csv('umbrella_idx.csv',index=False)
    so_df.to_csv('so_idx.csv', index=False)
    gen_df.to_csv('gender_idx.csv',index=False)
    return (um_df, so_df, gen_df)


def idx_grouping(start, end):
    tracking_start = start[0]
    pairs = []
    for i in range(len(start)):
        end_idx = i - 1
        if end_idx >= 0 and start[i] != end[end_idx] + 1:
            pairs.append([tracking_start, end[end_idx]])
            tracking_start = start[i]
    pairs.append([tracking_start, end[-1]])
    return pairs

if __name__ == '__main__':
   #identity_pipeline()
   get_ranges()