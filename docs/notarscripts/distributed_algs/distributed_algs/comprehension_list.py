squares = []

for i in range(10):
    squares.append(i**2)

print(squares)

list_comprehension = [i**2 for i in range(10)]

print(list_comprehension)


list_A = [1, 2, 3]
list_B = [3, 1, 4]

mix = []
for x in list_A:
    for y in list_B:
        if x != y:
            mix.append((x, y))

print(mix)

mix_list_comprehension = [(x, y) for x in list_A for y in list_B if x != y]

print(mix_list_comprehension)
