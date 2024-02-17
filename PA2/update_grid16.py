import json

with open('grid16_base.json', 'r') as infile:
    grid16 = json.load(infile)

for state in grid16['tran_prob']:
    if state == '0' or state == '15':
        continue

    if int(state) > 3:
        grid16['tran_prob'][state]['u'][int(state) - 4] = 1.0
    else:
        grid16['tran_prob'][state]['u'][int(state)] = 1.0

    if int(state) < 12:
        grid16['tran_prob'][state]['d'][int(state) + 4] = 1.0
    else:
        grid16['tran_prob'][state]['d'][int(state)] = 1.0

    if int(state) % 4 != 0:
        grid16['tran_prob'][state]['l'][int(state) - 1] = 1.0
    else:
        grid16['tran_prob'][state]['l'][int(state)] = 1.0

    if int(state) % 4 != 3:
        grid16['tran_prob'][state]['r'][int(state) + 1] = 1.0
    else:
        grid16['tran_prob'][state]['r'][int(state)] = 1.0

with open('grid16.json', 'w') as outfile:
    json.dump(grid16, outfile, indent=4)
