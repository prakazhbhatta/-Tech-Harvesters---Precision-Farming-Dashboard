import sys
import torch

if len(sys.argv) != 2:
    print('Usage: python inspect_pt.py <path_to_pt_file>')
    sys.exit(1)

pt_path = sys.argv[1]
obj = torch.load(pt_path, map_location=torch.device('cpu'))

print(f'Loaded object type: {type(obj)}')

if isinstance(obj, dict):
    print('Keys in the dict:')
    for k in obj.keys():
        print(f'  {k}')
    # If it looks like a state_dict, print some tensor shapes
    if all(isinstance(v, torch.Tensor) for v in obj.values()):
        print('\nTensor shapes:')
        for k, v in obj.items():
            print(f'  {k}: {tuple(v.shape)}')
    elif "state_dict" in obj:
        print('\nstate_dict keys:')
        for k in obj["state_dict"].keys():
            print(f'  {k}: {tuple(obj["state_dict"][k].shape)}')
else:
    print('Object attributes:')
    print(dir(obj)) 