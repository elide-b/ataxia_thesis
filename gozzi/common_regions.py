import xlrd
from openpyxl import load_workbook

# finding the structures present in Oh et al.
book = xlrd.open_workbook('oh_table1.xls')
sh = book.sheet_by_index(1)
structures_Oh = []

for value in sh.col_values(3):
    if isinstance(value, str):
        structures_Oh.append(value)

structures_Oh.pop(0)
print(f"Structures in Oh et al.: {structures_Oh}")

# finding the strucures present in Gozzi et al.
wb = load_workbook(filename='atlas_gozzi.xlsx')
sheet = wb.active
structures_Gozzi = []

for value in range(1, sheet.max_row):
    structure = sheet.cell(row=value, column=3)
    structures_Gozzi.append(structure.value)

structures_Gozzi.pop(0)
print(f"Structures in Gozzi et al.: {structures_Gozzi}")

# compare the two lists
common_regions = []

for s in structures_Gozzi:
    for r in structures_Oh:
        if s == r:
            common_regions.append(s)

print(f"Common regions between the two: {common_regions} \nFor a total of {len(common_regions)} common regions")
