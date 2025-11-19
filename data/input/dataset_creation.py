import identities
import os
from time import time

def create_dataset():
    header = "UUID,SystemIdentityIdx,ScenarioIdentityIdx,ScenarioIdx\n"
    with open('./data/input/dataset.csv','w') as fp:
        fp.write(header)

    if not os.path.isfile('./data/input/identities.txt'):
        umbrella, gender, so, ro = identities.get_queer_attributes()
        identities.save_identities_to_file(identities.attribute_pairing(umbrella, gender, so, ro))
    identity_list = []
    with open('./data/input/identities.txt','r') as fp:
        identity_list = fp.readlines()
    scenarios = ['TODO']*5
    buffer = []
    uuid = 0
    time_start = time()

    sys_idx = 0
    for _ in identity_list:
        scen_id_idx = 0
        for _ in identity_list:
            scen_idx = 0
            for _ in scenarios:
                buffer.append(f"{uuid},{sys_idx},{scen_id_idx},{scen_idx}\n")
                scen_idx +=1

                if len(buffer) >= 100:
                    with open('./data/input/dataset.csv','a') as fp:
                        fp.writelines(buffer)
                    buffer = []
                if uuid%1000 == 0:
                    progress = ((10*uuid)//(len(identity_list)**2 * len(scenarios)))
                    time_update = time() - time_start
                    time_update = str(int(time_update//60)) + 'min ' + str(int(time_update%60)) + 's'
                    print("[" + "="*progress + '>' + ' '*(10 - progress) + '] ' + str(progress*10) + "% " + time_update,end='\r')
                uuid += 1
            scen_id_idx += 1
        sys_idx += 1

if __name__ == '__main__':
    create_dataset()