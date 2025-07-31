import pickle

path='/home/pc-igp-173/Documentos/DATA/temporal/trails_JUN_2024_Day2-2.pickle'
output_files = ['/home/pc-igp-173/Documentos/DATA/temporal/trails_JUN_2024_Day2-2a.pickle', '/home/pc-igp-173/Documentos/DATA/temporal/trails_JUN_2024_Day2-2b.pickle', '/home/pc-igp-173/Documentos/DATA/temporal/trails_JUN_2024_Day2-2c.pickle']
n=0
n1 = n2 = n3 = 0
with open(path, 'rb') as f:
    while True:
        try:
            obj = pickle.load(f)
            del obj
            n+=1
            print(f"\rTrail: {n:<9}", end='', flush=True)
        except EOFError:
            break



total = n
print(f"Total objects found: {total}")

third = total // 3
limits = [third, 2 * third, total]

# Re-open input and output files
with open(path, 'rb') as f:
    for i in range(total):
        try:
            obj = pickle.load(f)

            if i < limits[0]:
                target = 0
                n1+=1
                n_i=n1
            elif i < limits[1]:
                target = 1
                n2+=1
                n_i=n2
            else:
                target = 2
                n3 += 1
                n_i=n3
            
            print(f"\rTrail: {n_i:<9} to target {target}", end='', flush=True)
            with open(output_files[target], 'ab') as out:
                pickle.dump(obj, out)

        except EOFError:
            print("Unexpected end of file.")
            break

print(f"\nObjects written to {output_files[0]}: {n1}")
print(f"Objects written to {output_files[1]}: {n2}")
print(f"Objects written to {output_files[2]}: {n3}")