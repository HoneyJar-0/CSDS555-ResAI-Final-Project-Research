from configs import experiment_config

def get_queer_attributes():
    '''
    Returns categorized identity attributes obtained from the gender census

    *Disclaimer: We are aware that some of these identities could belong in other categories. For the sake of simplicity
    and proof-of-concept, we will use these concrete categories. Further experimentation can involve overlapping.
    '''
    umbrella = ['','queer','LGBTQIA+'] #not included but used: nonqueer
    gender   = ['','cisgender','man','woman','nonbinary','gender non-conforming','agender','genderfluid','transgender','transwoman','transman','androgynous', 'genderqueer', 'intersex']
    so       = ['','gay','lesbian','bisexual','pansexual','straight','asexual','demisexual','homosexual','heterosexual']
    ro       = ['','homoromantic','heteroromantic','biromantic','panromantic','aromantic']
    
    return umbrella, gender, so, ro

def attribute_pairing(umbrella, gender, so, ro):
    permutations = []
    for u in umbrella: 
        for s in so:
            for r in ro:
                for g in gender:
                    do_not_append_person = ['man','woman'] #grammar; "I am a man person" is not grammatically correct
                    if not any(substring in g for substring in do_not_append_person):
                        g = (g + ' person') if g != '' else 'person'

                    identity = ""
                    for attr in [u,s,r,g]:
                        if attr != '':
                            identity = identity + attr + ' '
                    # Now we filter out nonqueer identities from the permutations.
                    if not ((s == 'straight' or s == 'heterosexual') and r == 'heteroromantic' and (g == 'man'  or g == 'woman')):
                        permutations.append({
                            "identity": identity.strip(),
                            "umbrella": u if u else None,
                            "gender": g if g else None,
                            "sexual_orientation": s if s else None,
                            "romantic_orientation": r if r else None
                        })
    for gender in ['man','woman','person']:
        permutations.append({
            "identity": 'nonqueer ' + gender,
            "umbrella": 'nonqueer',
            "gender": gender,
            "sexual_orientation": None,
            "romantic_orientation": None
        })
    return permutations


def identity_pipeline():
    umbrella, gender, so, ro = get_queer_attributes()
    print(f"Number entries:\nUmbrella: {len(umbrella)}, Gender: {len(gender)}, SO: {len(so)}, RO: {len(ro)}")
    permutations = attribute_pairing(umbrella, gender, so, ro)
    print(f"Number of Identities: {len(permutations)}")
    print(f"Number of dataset entries: {len(permutations)*len(permutations)}")
    
    # Add id to each permutation
    rows = []
    for i, p in enumerate(permutations):
        rows.append({"id": i, **p})
    return rows

if __name__ == '__main__':
    rows = identity_pipeline()
    print(f"Generated {len(rows)} identities")
    for r in rows[:5]:
        print(r)
