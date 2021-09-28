import Main_negation

'''
Check how often the program learns the different target predicates
'''
p_values = [0.1, 0.15]
#p_values = [0.5]
'''
# Innocent
with open("mislabelled_data.txt", "a") as f:        
    f.write("innocent: ---------------\n")           
for p in p_values:                                  
    for i in range(10):
        loss = Main_negation.innocent(p)             
        with open("mislabelled_data.txt", "a") as f:
            f.write(f"{i}, p-value={p}: {loss}\n")

# can_fly
with open("mislabelled_data.txt", "a") as f:
    f.write("can_fly: ---------------\n")
for p in p_values:
    for i in range(10):
        loss = Main_negation.can_fly(p)
        with open("mislabelled_data.txt", "a") as f:
            f.write(f"{i}, p-value={p}: {loss}\n")

# even
with open("mislabelled_data.txt", "a") as f:
    f.write("even: ---------------\n")
for p in p_values:
    for i in range(0, 10):
        loss = Main_negation.even_numbers_negation_test(p)
        with open("mislabelled_data.txt", "a") as f:
            f.write(f"{i}, p-value={p}: {loss}\n")

# has_roommate
with open("mislabelled_data.txt", "a") as f:
    f.write("has_roommate: ---------------\n")
for p in p_values:
    for i in range(0, 10):
        loss = Main_negation.has_roommate(p)
        with open("mislabelled_data.txt", "a") as f:
            f.write(f"{i}, p-value={p}: {loss}\n")

# two_children
with open("mislabelled_data.txt", "a") as f:
    f.write("two_children: ---------------\n")
for p in p_values:
    for i in range(10):
        loss = Main_negation.two_children(p)
        with open("mislabelled_data.txt", "a") as f:
            f.write(f"{i}, p-value={p}: {loss}\n")
'''
# not_grandparent
with open("mislabelled_data.txt", "a") as f:
    f.write("not_grandparent: ---------------\n")
for p in p_values:
    for i in range(5, 10):
        loss = Main_negation.not_grandparent(p)
        with open("mislabelled_data.txt", "a") as f:
            f.write(f"{i}, p-value={p}: {loss}\n")
