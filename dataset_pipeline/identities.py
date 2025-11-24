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
    unsorted = 'queer, LGBTQIA+, cisgender, man, woman, male, female, gender conforming, nonbinary, enby, gender non-conforming, polygender, agender, genderless, genderfluid, xenogender, transgender, transsexual, trans, transwoman, transman, genderqueer, pangender, demigender, intersexual, intersex, androgynous, gay, lesbian, bisexual, pansexual, straight, heterosexual, homosexual, asexual, demisexual, homoromantic, biromantic, panromantic, aromantic, heteroromantic'
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
        elif attribute == 'demisexual':
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
    with open("./data/input/identities.csv", 'w') as fp:
        fp.write("id,identity\n")
        for i, ident in enumerate(identities):
            fp.write(f"{i},{ident}\n")


if __name__ == '__main__':
    umbrella, gender, so, ro = get_queer_attributes()
    print(f"Number entries:\nUmbrella: {len(umbrella)}, Gender: {len(gender)}, SO: {len(so)}, RO: {len(ro)}")
    print(umbrella, gender, so, ro)
    permutations = attribute_pairing(umbrella, gender, so, ro)
    save_identities_to_file(permutations)