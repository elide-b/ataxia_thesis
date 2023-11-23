


found_list = []
voxel_count = []
rows_to_be_saved = []
regions_missing = []
cc = 0

for j in brain.region_labels:  # go over the region labels
    j = j.split(" ", 1)[1]
    # let us get rid of the left or right
    a = len(found_list)
    for i in sheet_data:
        if i[3] == j:  # or i[2] == "string2" or i[2] == "string3" or i[2] == "string4" or i[2] == "string5":
            found_list.append(i[4])
            if i[5]:
                voxel_count.append(int(i[5]))
            else:
                voxel_count.append(-1)
        else:
            rows_to_be_saved.append(i)

    b = len(found_list)
    if a == b:
        regions_missing.append(j)
        found_list.append('X')
        voxel_count.append(-1)
        print(a, b)

    b = len(found_list)
    cc += 1
    if b != cc:
        print(b, cc)
print("Regions missing:\n%s" % str(regions_missing))
n_regs = len(found_list)
print("Number of regions: %d" % n_regs)
n_regs2 = n_regs / 2
major_structures_labels = ["Right " + msl if iL < n_regs2 else "Left " + msl
                           for iL, msl in enumerate(found_list)]
voxel_count = np.array(voxel_count).astype('i')
major_structures = np.unique(major_structures_labels)
print("\nmajor_structures:\n", major_structures)