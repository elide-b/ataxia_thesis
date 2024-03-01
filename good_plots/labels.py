import xlrd

from connectivity import load_mousebrain

brain = load_mousebrain("Connectivity_596.h5", norm="log", scale="region")

# import the sheet
Oh_files = xlrd.open_workbook('oh_table1.xls')
sheet = Oh_files[1]
Oh_structures = []

for rownum in range(sheet.nrows):
    Oh_structures.append(sheet.cell_value(rownum, 3))
    # Oh_structures[1].append(sheet.cell_value(rownum, 4))

print(Oh_structures)

structures_found = []
index_found = []

# iter over the rows to find names in brain.region_labels
for i, r in enumerate(Oh_structures):
    for n in brain.region_labels:
        n = n.split(" ", 1)[1]  # get rid of the left or right
        if n == r:
            structures_found.append(n)
            index_found.append()

print(structures_found)
print(index_found)
exit()

# find the regions in the connectivity
nreg = len(brain.region_labels)
region_found = [i for i in range(nreg) if any(b in Oh_name.iterrows() for b in brain.region_labels)]

print(region_found)
