import Main_negation

'''
Check how often the program learns the different target predicates
'''
'''
# Innocent
losses = []
for i in range(25):
    loss = Main_negation.innocent()
    losses.append(loss)

with open("learn_precentage.txt", "a") as f:
    f.write("Innocent: ---------------\n")
    for index, loss in enumerate(losses):
        f.write(f"{index}: {loss}\n")

# can_fly
losses = []
for i in range(15):
    loss = Main_negation.can_fly()
    losses.append(loss)

with open("learn_precentage.txt", "a") as f:
    f.write("can_fly: ---------------\n")
    for index, loss in enumerate(losses):
        f.write(f"{index}: {loss}\n")


# even
losses = []
for i in range(10):
    loss = Main_negation.even_numbers_negation_test()
    losses.append(loss)

with open("learn_precentage.txt", "a") as f:
    f.write("even: ---------------\n")
    for index, loss in enumerate(losses):
        f.write(f"{index}: {loss}\n")


# has_roommate
with open("learn_precentage.txt", "a") as f:
    f.write("has_roommate: ---------------\n")
    for index in range(10):
        loss = Main_negation.has_roommate()
        f.write(f"{index}: {loss}\n")


# two_children
losses = []
for i in range(10):
    loss = Main_negation.two_children()
    losses.append(loss)

    with open("learn_precentage.txt", "a") as f:
        f.write(f"{i}: {loss}\n")
'''

# not_grandparent
losses = []
for i in range(10):
    loss = Main_negation.not_grandparent()
    losses.append(loss)

    with open("learn_precentage.txt", "a") as f:
        f.write(f"{i}: {loss}\n")

