import identities
import os
from time import time

def create_dataset():
    header = "UUID,System_Identity,Scenario_Identity,Prompt\n"
    with open('./dataset.csv','w') as fp:
        fp.write(header)

    if not os.path.isfile('./data/input/identities.txt'):
        umbrella, gender, so, ro = identities.get_queer_attributes()
        identities.save_identities_to_file(identities.attribute_pairing(umbrella, gender, so, ro))
    identity_list = []
    with open('./data/input/identities.txt','r') as fp:
        identity_list = fp.readlines()
    with open('./data/input/scenarios.txt','r') as fp:
        scenarios = fp.readlines()
    buffer = []
    uuid = 0
    time_start = time()
    for sys_identity in identity_list:
        for scen_identity in identity_list:
            for scenario in scenarios:
                buffer.append(f"{uuid},{sys_identity.strip()},{scen_identity.strip()},{prompt_creation(sys_identity, scen_identity, scenario)}\n")
                if len(buffer) >= 50:
                    with open('./dataset.csv','a') as fp:
                        fp.writelines(buffer)
                    buffer = []
                if uuid%1000 == 0:
                    progress = ((10*uuid)//(len(identity_list)**2 * len(scenarios)))
                    time_update = time() - time_start
                    time_update = str(int(time_update//60)) + 'min ' + str(int(time_update%60)) + 's'
                    print("[" + "="*progress + '>' + ' '*(10 - progress) + '] ' + str(progress*10) + "% " + time_update,end='\r')
                uuid += 1
def prompt_creation(a,b,c):
    #header, TODO
    return "TODO"

if __name__ == '__main__':
    create_dataset()