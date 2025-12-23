from pygbx import Gbx, GbxType

g = Gbx('rl_lvl_1.Map.Gbx')
challenges = g.get_classes_by_ids([GbxType.CHALLENGE, GbxType.CHALLENGE_OLD])
if not challenges:
    quit()

challenge = challenges[0]
for block in challenge.blocks:
    print(block)

import pandas as pd

data = []

for block in challenge.blocks:
    print(block.position.x)
    data.append({
        'Block': block.name,              # or block.model.name depending on pygbx version
        'X': block.position.x,
        'Y': block.position.y,
        'Z': block.position.z,
        'Dir': block.rotation
    })

df = pd.DataFrame(data, columns=['Block', 'X', 'Y', 'Z', 'Dir'])
print(df.columns)
df.to_csv('clean_blocks.csv', index=False)
print("Saved to clean_blocks.csv")

